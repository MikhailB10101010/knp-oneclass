/**
 * @file oneclass_runner.cpp
 * @brief One-class SNN runner for KNP backends (CPU / AltAI-2 emulator).
 *
 * Executes an RBF-like one-class spiking network:
 *   input (N features, rate-coded) -> hidden (K AltAILIF prototype neurons)
 * and prints the total hidden-population spike count per sample (score).
 *
 * Usage:
 *   oneclass_runner <backend.so> <prototypes.csv> <samples.csv> <steps> <activation_threshold>
 *
 *   prototypes.csv : K rows x N cols, integer weights in [-255, 255]
 *   samples.csv    : M rows x N cols, floats in [0, 1]
 *
 * Exit code 0 on success; scores are printed to stdout, one per line.
 */

#include <knp/core/messaging/messaging.h>
#include <knp/framework/backend_loader.h>
#include <knp/framework/model.h>
#include <knp/framework/model_executor.h>
#include <knp/framework/network.h>
#include <knp/neuron-traits/altai_lif.h>
#include <knp/synapse-traits/delta.h>

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>


using AltAILIFPopulation = knp::core::Population<knp::neuron_traits::AltAILIF>;
using DeltaProjection = knp::core::Projection<knp::synapse_traits::DeltaSynapse>;


/// Read a CSV file (comma-separated, no header) into a row-major matrix of doubles.
static std::vector<std::vector<double>> read_csv(const std::string &path)
{
    std::ifstream in(path);
    if (!in) throw std::runtime_error("Cannot open file: " + path);
    std::vector<std::vector<double>> rows;
    std::string line;
    while (std::getline(in, line))
    {
        if (line.empty()) continue;
        std::vector<double> row;
        std::stringstream ss(line);
        std::string cell;
        while (std::getline(ss, cell, ',')) row.push_back(std::stod(cell));
        rows.push_back(std::move(row));
    }
    return rows;
}


/// Run one sample through the network on the given backend; return total hidden spikes.
static size_t run_sample(
    knp::framework::BackendLoader &loader, const std::string &backend_path,
    const std::vector<std::vector<int>> &weights,  // K x N quantized prototype weights
    const std::vector<double> &sample,             // N floats in [0, 1]
    size_t steps, uint16_t activation_threshold)
{
    const size_t n_in = sample.size();
    const size_t n_hidden = weights.size();

    // Neuron parameters: strict LIF, no leak, hard reset.
    auto neuron_generator = [activation_threshold](size_t)
    {
        knp::neuron_traits::neuron_parameters<knp::neuron_traits::AltAILIF> p{};
        p.activation_threshold_ = activation_threshold;
        p.negative_activation_threshold_ = 0;
        p.potential_leak_ = 0;
        p.is_reset_ = true;
        p.potential_reset_value_ = 0;
        return p;
    };

    AltAILIFPopulation population{neuron_generator, n_hidden};

    // Dense input projection: synapse (n, k) carries weight of prototype k, feature n.
    auto synapse_generator = [&weights, n_in](size_t index) -> std::optional<DeltaProjection::Synapse>
    {
        const size_t n = index % n_in;
        const size_t k = index / n_in;
        const int w = weights[k][n];
        const auto type = w >= 0 ? knp::synapse_traits::OutputType::EXCITATORY
                                 : knp::synapse_traits::OutputType::INHIBITORY_CURRENT;
        const float weight = static_cast<float>(w >= 0 ? w : -w);
        if (weight == 0.f) return std::nullopt;
        const knp::synapse_traits::synapse_parameters<knp::synapse_traits::DeltaSynapse> params{
            weight, 1, type};
        return std::make_tuple(params, n, k);
    };

    DeltaProjection input_projection{
        knp::core::UID{false}, population.get_uid(), synapse_generator, n_in * n_hidden};

    knp::core::UID input_uid = input_projection.get_uid();
    knp::core::UID output_uid = population.get_uid();

    knp::framework::Network network;
    network.add_population(std::move(population));
    network.add_projection(std::move(input_projection));

    knp::core::UID i_channel_uid, o_channel_uid;
    knp::framework::Model model(std::move(network));
    model.add_input_channel(i_channel_uid, input_uid);
    model.add_output_channel(o_channel_uid, output_uid);

    // Deterministic rate coding: feature v in [0,1] -> round(v*steps) spikes,
    // evenly spread over the simulation window.
    std::vector<std::vector<size_t>> spike_times(n_in);
    for (size_t n = 0; n < n_in; ++n)
    {
        const size_t count = static_cast<size_t>(sample[n] * steps + 0.5);
        for (size_t j = 0; j < count; ++j) spike_times[n].push_back(j * steps / (count ? count : 1));
    }
    auto input_gen = [&spike_times](knp::core::Step step) -> knp::core::messaging::SpikeData
    {
        knp::core::messaging::SpikeData data;
        for (size_t n = 0; n < spike_times.size(); ++n)
            for (size_t t : spike_times[n])
                if (t == step) data.push_back(n);
        return data;
    };

    auto backend = loader.load(backend_path);
    knp::framework::ModelExecutor executor(model, backend, {{i_channel_uid, input_gen}});
    auto &out_channel = executor.get_loader().get_output_channel(o_channel_uid);

    executor.start([steps](size_t step) { return step < steps; });

    size_t total_spikes = 0;
    for (const auto &msg : out_channel.update()) total_spikes += msg.neuron_indexes_.size();
    return total_spikes;
}


int main(int argc, char **argv)
{
    if (argc != 6)
    {
        std::cerr << "Usage: oneclass_runner <backend.so> <prototypes.csv> <samples.csv> <steps> <activation_threshold>\n";
        return 1;
    }
    try
    {
        const std::string backend_path = argv[1];
        const auto protos_d = read_csv(argv[2]);
        const auto samples = read_csv(argv[3]);
        const size_t steps = std::stoul(argv[4]);
        const auto threshold = static_cast<uint16_t>(std::stoul(argv[5]));

        std::vector<std::vector<int>> weights(protos_d.size());
        for (size_t k = 0; k < protos_d.size(); ++k)
            for (double w : protos_d[k]) weights[k].push_back(static_cast<int>(w));

        knp::framework::BackendLoader loader;
        for (const auto &sample : samples)
            std::cout << run_sample(loader, backend_path, weights, sample, steps, threshold) << "\n";
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << "\n";
        return 2;
    }
    return 0;
}

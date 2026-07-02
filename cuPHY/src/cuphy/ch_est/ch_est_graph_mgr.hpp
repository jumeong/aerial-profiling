/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef CUPHY_CH_EST_GRAPH_MGR_HPP
#define CUPHY_CH_EST_GRAPH_MGR_HPP

#include <vector>
#include <cstdint>
#include <stdexcept>

#include "fmt/format.h"

#include <gsl-lite/gsl-lite.hpp>

#include "cuphy.h"
#include "ch_est_utils.hpp"
#include "IGraph_mgr.hpp"

namespace ch_est {

/**
 * @brief ChestSubSlotNodes for ChestEarlyHarqNodes/ChestFrontDmrsNodes is managing
 *        `CUGraphNode`s, bookkeeping of enable/disable nodes in the graph.
 * It is being used by class ChannelEstimateGraphMgr and class PuschRx is using it via
 * the manager class.
 * - Add kernel/secondary kernel
 * - Set enable/disable - aka status
 */
class ChestSubSlotNodes final : public IChestSubSlotNodes {
public:
    /**
     * @param nMaxChEstHetCfgs Max Channel estimate Cfgs
     * @param cfgs0Slot single config, from 0 slot.
     */
    explicit ChestSubSlotNodes(const std::uint32_t nMaxChEstHetCfgs,
                               const cuphyPuschRxChEstLaunchCfgs_t &cfgs0Slot,
                               const cuphyPuschChEstAlgoType_t chEstAlgo) :
            m_chEstAlgo{chEstAlgo},
            m_nMaxChEstHetCfgs{nMaxChEstHetCfgs},
            m_cfgs0Slot{cfgs0Slot},
            m_nodesEnabled(m_nMaxChEstHetCfgs, std::numeric_limits<uint8_t>::max()),
            m_secondNodesEnabled(m_nodesEnabled) {
    }

    ChestSubSlotNodes(const ChestSubSlotNodes &chestEarlyHarqNodes) = delete;
    ChestSubSlotNodes &operator=(const ChestSubSlotNodes &chestEarlyHarqNodes) = delete;

    /**
     * @brief Given a graph, add the internal kernel nodes to the graph.
     * @param graph CUDA graph
     * @param currNodeDeps Current dependencies to use when adding the next node
     * @param nextNodeDeps The dependencies for the next node that will be added to the graph
     * in subsequent calls.
     * @param nodeParams Parameters to be used for the KERNEL that is added.
     */
    void addKernelNodeToGraph(CUgraph graph,
                              std::vector<CUgraphNode> &currNodeDeps,
                              std::vector<CUgraphNode> &nextNodeDeps,
                              CUDA_KERNEL_NODE_PARAMS &nodeParams) final;

    /**
     * @brief Given a graph, add the internal kernel nodes to the graph.
     * This is the secondary kernel of Channel Estimate
     * @param graph CUDA graph
     * @param currNodeDeps Current dependencies to use when adding the next node
     * @param nextNodeDeps The dependencies for the next node that will be added to the graph
     * in subsequent calls.
     * @param nodeParams Parameters to be used for the KERNEL that is added.
     */
    void addSecondaryKernelNodeToGraph(CUgraph graph,
                                       std::vector<CUgraphNode> &currNodeDeps,
                                       std::vector<CUgraphNode> &nextNodeDeps,
                                       CUDA_KERNEL_NODE_PARAMS &nodeParams) final;

    /**
     * @brief set node status on the primary kernel, enable disable nodes
     * @param disableAllNodes if marked as disable all node, all nodes are disabled
     * @param graphExec the CUDA graph exec to use
     */
    void setNodeStatus(ChestCudaUtils::DisableAllNodes disableAllNodes, CUgraphExec graphExec) final;

    /**
     * @brief set node status on the secondary kernel, enable disable nodes
     * @param disableAllNodes if marked as disable all node, all nodes are disabled
     * @param graphExec the CUDA graph exec to use
     */
    void setSecondaryNodeStatus(ChestCudaUtils::DisableAllNodes disableAllNodes, CUgraphExec graphExec) final;

private:
    cuphyPuschChEstAlgoType_t m_chEstAlgo{};
    std::uint32_t m_nMaxChEstHetCfgs{};
    const cuphyPuschRxChEstLaunchCfgs_t &m_cfgs0Slot; // single cfg
    CUgraphNode m_nodes[CUPHY_PUSCH_RX_CH_EST_ALL_ALGS_N_MAX_HET_CFGS]{},
            m_secondNodes[CUPHY_PUSCH_RX_CH_EST_ALL_ALGS_N_MAX_HET_CFGS]{};
    std::vector<uint8_t> m_nodesEnabled, m_secondNodesEnabled;
};

/**
 * @brief ChestNodes is managing `CUGraphNode`s, bookkeeping of enable/disable nodes
 *        in the graph.
 *
 * It is being used by class ChannelEstimateGraphMgr and class PuschRx is using it via
 * the manager class.
 * - Add kernel/secondary kernel
 * - Set enable/disable - aka status
 */
class ChestNodes final : public IChestGraphNodes {
public:
    /**
     * @brief Channel Estimate nodes construction
     * @param nMaxChEstHetCfgs Maximum of Channel estimate heterogeneous configs.
     * @param earlyHarqModeEnabled Early HARQ enable/disable
     * @param chEstAlgo The algorithm used for channel estimate
     */
    ChestNodes(const std::uint32_t nMaxChEstHetCfgs,
               const bool earlyHarqModeEnabled,
               const cuphyPuschChEstAlgoType_t chEstAlgo) :
            m_nMaxChEstHetCfgs{nMaxChEstHetCfgs},
            m_earlyHarqModeEnabled{earlyHarqModeEnabled},
            m_chEstAlgo{chEstAlgo},
            m_chEstNodesEnabled(CUPHY_PUSCH_RX_MAX_N_TIME_CH_EST, std::vector<uint8_t>(m_nMaxChEstHetCfgs,
                                                                                       std::numeric_limits<uint8_t>::max())),
            m_chEstSecondNodesEnabled(m_chEstNodesEnabled) {
        if (!nMaxChEstHetCfgs) {
            throw std::invalid_argument(std::string(__func__) + " nMaxChEstHetCfgs cannot be zero");
        }
    }

    ChestNodes(const ChestNodes &chestNodes) = delete;
    ChestNodes &operator=(const ChestNodes &chestNodes) = delete;

    /**
     * @brief return a view to the channel estimate Kernel Launch Configs
     * @return view to the channel estimate kernel launch configs
     */
    [[nodiscard]] auto chEstLaunchCfgs() noexcept { return gsl_lite::span(m_chEstLaunchCfgs); }

    /**
     * @brief First channel estimate launch configs. Some classes just need the first one.
     * @return A reference to the first channel estimate kernel launch configs.
     */
    [[nodiscard]] const auto &chEstFirstLaunchCfgs() const noexcept { return m_chEstLaunchCfgs[0]; }

    /**
     * @brief initialize all number of configs to 0.
     */
    void init() final {
        for (auto &m_chEstLaunchCfg: m_chEstLaunchCfgs) {
            m_chEstLaunchCfg.nCfgs = 0;
        }
    }

    /**
     * @brief Set boolean of early HARQ
     * @param earlyHarqModeEnabled true/false
     */
    void setEarlyHarqModeEnabled(const bool earlyHarqModeEnabled) final {
        m_earlyHarqModeEnabled = earlyHarqModeEnabled;
    }

    /**
     * @brief Given a graph, add the internal kernel nodes to the graph.
     * @param graph CUDA graph
     * @param currNodeDeps Current dependencies to use when adding the next node
     * @param nextNodeDeps The dependencies for the next node that will be added to the graph
     * in subsequent calls.
     * @param nodeParams Parameters to be used for the KERNEL that is added.
     */
    void addKernelNodeToGraph(CUgraph graph,
                              std::vector<CUgraphNode> &currNodeDeps,
                              std::vector<CUgraphNode> &nextNodeDeps,
                              CUDA_KERNEL_NODE_PARAMS &nodeParams) final;

    /**
     * @brief Given a graph, add the internal kernel nodes to the graph.
     * This is the secondary kernel of Channel Estimate
     * @param graph CUDA graph
     * @param currNodeDeps Current dependencies to use when adding the next node
     * @param nextNodeDeps The dependencies for the next node that will be added to the graph
     * in subsequent calls.
     * @param nodeParams to be used for the KERNEL that is added.
     */
    void addSecondaryKernelNodeToGraph(CUgraph graph,
                                       std::vector<CUgraphNode> &currNodeDeps,
                                       std::vector<CUgraphNode> &nextNodeDeps,
                                       CUDA_KERNEL_NODE_PARAMS &nodeParams) final;

    /**
     * @brief set node status on the primary kernel, enable disable nodes
     * @param disableAllNodes if marked as disable all node, all nodes are disabled
     * @param graphExec the CUDA graph exec to use
     */
    void setNodeStatus(ChestCudaUtils::DisableAllNodes disableAllNodes, CUgraphExec graphExec) final;

    /**
     * @brief set node status on the secondary kernel, enable disable nodes
     * @param disableAllNodes if marked as disable all node, all nodes are disabled
     * @param graphExec the CUDA graph exec to use
     */
    void setSecondaryNodeStatus(ChestCudaUtils::DisableAllNodes disableAllNodes, CUgraphExec graphExec) final;

    /**
     * @brief disable nodes in slot0
     * @param graphExec CUDA graph exec
     */
    void disableNodes0Slot(CUgraphExec graphExec) final;

private:
    std::uint32_t m_nMaxChEstHetCfgs{};
    bool m_earlyHarqModeEnabled{};
    cuphyPuschChEstAlgoType_t m_chEstAlgo{};
    cuphyPuschRxChEstLaunchCfgs_t m_chEstLaunchCfgs[CUPHY_PUSCH_RX_MAX_N_TIME_CH_EST]{};
    CUgraphNode m_chEstNodes[CUPHY_PUSCH_RX_MAX_N_TIME_CH_EST][CUPHY_PUSCH_RX_CH_EST_ALL_ALGS_N_MAX_HET_CFGS]{};
    CUgraphNode m_chEstSecondNodes[CUPHY_PUSCH_RX_MAX_N_TIME_CH_EST][CUPHY_PUSCH_RX_CH_EST_ALL_ALGS_N_MAX_HET_CFGS]{};
    std::vector<std::vector<uint8_t>> m_chEstNodesEnabled, m_chEstSecondNodesEnabled;
};

/**
 * @brief ChannelEstimateGraphMgr is a top-level class that is managing all graph creation,
 *        enable/disable of graph nodes, delegating it to the corresponding data members which
 *        do the actual work.
 *
 * It is used by @class PuschRx. It will call the accessors to obtain access to the relevant interface
 * of each Channel estimate Node collection handler (class ChestNodes, class ChestEarlyHarqNodes,
 * class ChestFrontDmrsNodes)
 */
class ChannelEstimateGraphMgr final {
public:
    // need to pass early-harq mode so ChestNodes can be used
    explicit ChannelEstimateGraphMgr(const std::uint32_t nMaxChEstHetCfgs,
                                     const bool earlyHarqModeEnabled,
                                     const cuphyPuschChEstAlgoType_t chEstAlgo) :
            m_chestNodes(nMaxChEstHetCfgs, earlyHarqModeEnabled, chEstAlgo),
            m_chestEarlyHarqNodes(nMaxChEstHetCfgs,
                                  m_chestNodes.chEstFirstLaunchCfgs(),
                                  chEstAlgo),
            m_chestFrontDmrsNodes(nMaxChEstHetCfgs,
                                  m_chestNodes.chEstFirstLaunchCfgs(),
                                  chEstAlgo) {
        if (!nMaxChEstHetCfgs) {
            throw std::invalid_argument(fmt::format("{} nMaxChEstHetCfgs cannot be zero", __func__));
        }
    }

    /**
     * @brief Set/toggle true/false of Early HARQ mode.
     *  Call chain related to the above:
     *  PuschRx::setup();
     *    setupCmnPhase1();
     *    setupCmnPhase2();
     * @param earlyHarqModeEnabled true/false
     */
    void setEarlyHarqModeEnabled(const bool earlyHarqModeEnabled) {
        m_chestNodes.setEarlyHarqModeEnabled(earlyHarqModeEnabled);
    }

    /**
     * @brief Expose pointer to the Channel Estimate Launch Configs
     * @return non owning view/span of Channel estimate kernel launch configs.
     */
    [[nodiscard]]
    auto getLaunchCfgs() { return m_chestNodes.chEstLaunchCfgs(); }

    /**
     * Return the types references as is to the channel estimate specific
     * class type that will expose interfaces to these classes.
     *
     * This interfaces in channel estimate are defined in abstract base
     * class IModule.
     */
    [[nodiscard]] auto& asChest() { return m_chestNodes; }
    [[nodiscard]] auto& asEarlyHarq() { return m_chestEarlyHarqNodes; }
    [[nodiscard]] auto& asFrontDmrs() { return m_chestFrontDmrsNodes; }

private:
    ChestNodes m_chestNodes;
    ChestSubSlotNodes m_chestEarlyHarqNodes;
    ChestSubSlotNodes m_chestFrontDmrsNodes;
};

} // namespace ch_est

#endif //CUPHY_CH_EST_GRAPH_MGR_HPP

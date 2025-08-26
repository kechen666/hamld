from typing import List, Set, Dict, Tuple
import logging
from concurrent.futures import ThreadPoolExecutor
from hamld.logging_config import setup_logger
from hamld.contraction_strategy.hypergraph_to_connectivity import ConnectivityGraph

# Set up logging configuration
logger = setup_logger("./contraction_strategy/mld_order_finder", logging.WARNING)


class GreedyMLDOrderFinder:
    """
    A class to find the greedy MLD contraction order by iteratively contracting nodes with
    the minimum degree or the smallest change in the probability distribution dimension.
    """

    def __init__(self, connectivity_graph: ConnectivityGraph):
        self.connectivity_graph = connectivity_graph
        self.max_prob_dist_dimension = 0
        self.max_prob_dist_dimension_count = 0
        self.max_contracted_nodes: List[str] = []
        self.order: List[str] = []

        # Initialize state variables
        self.all_nodes: Set[str] = set(self.connectivity_graph.nodes())
        # 当前收缩的概率分布维度
        self.current_prob_dist_dimension: int = 0
        # 当前收缩的节点集合
        self.current_contracted_nodes: Set[str] = set()
        # 当前收缩的节点的相关节点集合
        self.current_prob_dist_related_nodes: Set[str] = set()
        # 当前收缩的节点的相邻候选节点集合
        self.current_candidate_nodes_for_contraction: Set[str] = set()

    def find_order(self) -> List[str]:
        """
        Find the greedy MLD contraction order by iterating over each step to contract nodes.

        Returns:
            List[str]: The order of contracted nodes.
        """
        nodes_number = self.connectivity_graph.number_of_nodes()

        # Iterate over each contraction step
        for step in range(nodes_number):
            logger.debug(f"Step: {step}")

            # First step: select node with minimum degree
            if step == 0:
                detector = min(self.connectivity_graph.nodes(), key=lambda node: self.connectivity_graph.degree(node))
                logger.debug(f"First detector: {detector}")
            else:
                # Subsequent steps: select node based on probability distribution change
                prob_dist_changes = self.compute_contraction_node_prob_dist_dim_change()
                logger.debug(f"Step: {step}, prob_dist_changes: {prob_dist_changes}")
                # Handle case when no candidate nodes available but nodes remain
                if not prob_dist_changes:  # 没有候选节点但仍有未收缩节点
                    remaining_nodes = self.all_nodes - self.current_contracted_nodes
                    if remaining_nodes:
                        detector = min(remaining_nodes, key=lambda node: self.connectivity_graph.degree(node))
                        logger.debug(f"No candidate nodes, selecting min degree from remaining: {detector}")
                    else:
                        break  # All nodes contracted, exit loop
                else:
                    # Select node with smallest probability distribution change
                    # If tie, choose node with smaller index (e.g. D3 before D5)
                    sorted_prob_dist_changes = sorted(prob_dist_changes.items(), key=lambda item: (item[1], item[0]))
                    detector = sorted_prob_dist_changes[0][0]
                    logger.debug(f"Step: {step}, prob_dist_changes: {prob_dist_changes}, selected detector: {detector}")

             # Record contracted node
            self.order.append(detector)
            logger.debug(f"Contracted detector (node): {detector}")

            # Update graph state after contraction
            self.contract_node_and_update_candidates(detector)

            # Calculate current probability distribution dimension
            self.current_prob_dist_dimension = len(self.current_prob_dist_related_nodes) - len(self.current_contracted_nodes)

            # Track maximum probability distribution dimension encountered
            if self.current_prob_dist_dimension > self.max_prob_dist_dimension:
                self.max_prob_dist_dimension = self.current_prob_dist_dimension
                self.max_prob_dist_dimension_count = 1
                self.max_contracted_nodes = [detector]
            elif self.current_prob_dist_dimension == self.max_prob_dist_dimension:
                self.max_prob_dist_dimension_count += 1
                self.max_contracted_nodes.append(detector)

            # Log current contraction state
            logger.debug(f"After contracting {detector}, current probability distribution dimension: {self.current_prob_dist_dimension}, size: {2 ** self.current_prob_dist_dimension}")
            logger.debug(f"Current contracted nodes: {self.current_contracted_nodes}")
            logger.debug(f"Current related nodes: {self.current_prob_dist_related_nodes}")
            logger.debug(f"Current candidate nodes for contraction: {self.current_candidate_nodes_for_contraction}")

        logger.debug(f"The contracted order is: {self.order}")
        return self.order

    def contract_node_and_update_candidates(self, detector: str):
        """
        Contract a node and update the related nodes set and candidate nodes for contraction.

        Args:
            detector (str): The node (detector) to be contracted.
        """
        # Add the detector and its neighbors to the related nodes set
        self.current_contracted_nodes.add(detector)
        self.current_prob_dist_related_nodes.add(detector)

        # Get the neighbors of the detector
        neighboring_nodes = self.connectivity_graph.neighbors(detector)

        # Update the related nodes set with the neighbors
        self.current_prob_dist_related_nodes.update(neighboring_nodes)

        # Update the candidate nodes by excluding the already contracted nodes
        self.current_candidate_nodes_for_contraction = self.current_prob_dist_related_nodes - self.current_contracted_nodes

    def compute_contraction_node_prob_dist_dim_change(self) -> Dict[str, int]:
        """ 
        Compute the change in the probability distribution dimension after contracting each node.

        Returns:
            Dict[str, int]: A dictionary mapping each candidate node to its change in the probability distribution dimension.
        """
        prob_dist_changes = {}

        for candidate_node in self.current_candidate_nodes_for_contraction:
            # Get the neighboring nodes of the candidate node
            neighboring_nodes = self.connectivity_graph.neighbors(candidate_node)

            # Determine the new nodes formed after contraction (exclude already contracted nodes)
            new_nodes_after_contraction = [
                node for node in neighboring_nodes
                if node not in self.current_contracted_nodes and node not in self.current_candidate_nodes_for_contraction
            ]

            # Record the change in probability distribution dimension
            prob_dist_changes[candidate_node] = len(new_nodes_after_contraction)

        return prob_dist_changes


class ParallelGreedyMLDOrderFinder(GreedyMLDOrderFinder):
    """
    A class to find the greedy MLD contraction order with parallelized computation for the 
    contraction node probability distribution change.
    """

    def __init__(self, connectivity_graph: ConnectivityGraph):
        super().__init__(connectivity_graph)

    def find_order(self) -> List[str]:
        """
        Find the greedy MLD contraction order by iterating over each step to contract nodes.
        Parallelize the computation of the contraction node probability distribution changes.

        Returns:
            List[str]: The order of contracted nodes.
        """
        nodes_number = self.connectivity_graph.number_of_nodes()

        # Iterate over each step (cannot be parallelized as each step depends on the previous one)
        for step in range(nodes_number):
            logger.debug(f"Step: {step}")

            # First step: find the node with minimum degree
            if step == 0:
                detector = min(self.connectivity_graph.nodes(), key=lambda node: self.connectivity_graph.degree(node))
                logger.debug(f"First detector: {detector}")
            else:
                # Subsequent steps: compute contraction probabilities in parallel
                with ThreadPoolExecutor() as executor:
                    prob_dist_changes = self.compute_contraction_prob_dist_dim_parallel(executor)
                detector = min(prob_dist_changes, key=prob_dist_changes.get)

            # Contract the node
            self.order.append(detector)
            logger.debug(f"Contracted detector (node): {detector}")

            # Contract the node and update related sets and candidate list
            self.contract_node_and_update_candidates(detector)

            # Compute the current probability distribution dimension
            self.current_prob_dist_dimension = len(self.current_prob_dist_related_nodes) - len(self.current_contracted_nodes)

            # Track the maximum probability distribution dimension
            if self.current_prob_dist_dimension > self.max_prob_dist_dimension:
                self.max_prob_dist_dimension = self.current_prob_dist_dimension
                self.max_prob_dist_dimension_count = 1
                self.max_contracted_nodes = [detector]
            elif self.current_prob_dist_dimension == self.max_prob_dist_dimension:
                self.max_prob_dist_dimension_count += 1
                self.max_contracted_nodes.append(detector)

            # Log the current state
            logger.debug(f"After contracting {detector}, current probability distribution dimension: {self.current_prob_dist_dimension}, size: {2 ** self.current_prob_dist_dimension}")
            logger.debug(f"Current contracted nodes: {self.current_contracted_nodes}")
            logger.debug(f"Current related nodes: {self.current_prob_dist_related_nodes}")
            logger.debug(f"Current candidate nodes for contraction: {self.current_candidate_nodes_for_contraction}")

        logger.debug(f"The contracted order is: {self.order}")
        return self.order

    def compute_contraction_prob_dist_dim_parallel(self, executor: ThreadPoolExecutor) -> Dict[str, int]:
        """ 
        Parallel version of the contraction probability distribution dimension computation.
        """
        # Submit tasks to executor to compute each candidate node's probability distribution change
        future_to_node = {
            executor.submit(self.compute_single_node_prob_dist_change, candidate_node): candidate_node
            for candidate_node in self.current_candidate_nodes_for_contraction
        }

        # Wait for all tasks to finish and collect results
        prob_dist_changes = {}
        for future in future_to_node:
            candidate_node = future_to_node[future]
            prob_dist_changes[candidate_node] = future.result()

        return prob_dist_changes

    def compute_single_node_prob_dist_change(
        self, 
        candidate_node: str
    ) -> int:
        """
        Compute the probability distribution dimension change for a single node.
        
        Args:
            candidate_node (str): The candidate node to compute the dimension change for.
        
        Returns:
            int: The change in probability distribution dimension for the candidate node.
        """
        # Get the neighboring nodes of the candidate node
        neighboring_nodes = self.connectivity_graph.neighbors(candidate_node)

        # Determine the new nodes formed after contraction (exclude already contracted nodes)
        new_nodes_after_contraction = [
            node for node in neighboring_nodes
            if node not in self.current_contracted_nodes and node not in self.current_candidate_nodes_for_contraction
        ]

        # Return the change in probability distribution dimension
        return len(new_nodes_after_contraction)

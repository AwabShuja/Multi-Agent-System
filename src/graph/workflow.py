"""
LangGraph Workflow Definition for the Multi-Agent Virtual Company.

This module creates and compiles the StateGraph that orchestrates
the multi-agent research workflow.

Workflow Flow:
    START -> supervisor -> [researcher|analyst|critic|writer] -> supervisor -> ... -> END
    
The supervisor acts as the central router, deciding which agent
should process next based on the current state.
"""

from typing import Optional, Callable
from datetime import datetime
from loguru import logger

from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.memory import MemorySaver

from src.graph.state import GraphState, create_initial_state, get_state_summary
from src.graph.nodes import (
    supervisor_node,
    researcher_node,
    analyst_node,
    critic_node,
    writer_node,
    end_node,
    initialize_registry,
    AgentRegistry,
)
from src.graph.edges import (
    route_from_supervisor,
    route_after_researcher,
    route_after_analyst,
    route_after_critic,
    route_after_writer,
)


# =============================================================================
# Workflow Builder
# =============================================================================

class WorkflowBuilder:
    """
    Builder class for creating the multi-agent workflow.
    
    Provides a fluent interface for configuring and building
    the LangGraph StateGraph.
    """
    
    def __init__(self, api_key: str):
        """
        Initialize the workflow builder.
        
        Args:
            api_key: Groq API key for agents
        """
        self.api_key = api_key
        self.graph: Optional[StateGraph] = None
        self.checkpointer: Optional[MemorySaver] = None
        
        # Initialize agent registry
        initialize_registry(api_key)
        
        logger.info("WorkflowBuilder initialized")
    
    def build(self) -> StateGraph:
        """
        Build the workflow graph.
        
        Creates all nodes and edges for the multi-agent workflow.
        
        Returns:
            Configured StateGraph (not yet compiled)
        """
        logger.info("Building workflow graph...")
        
        # Create StateGraph with our state schema
        self.graph = StateGraph(GraphState)
        
        # =================================================================
        # Add Nodes
        # =================================================================
        logger.debug("Adding nodes...")
        
        # Supervisor node - Central orchestrator
        self.graph.add_node("supervisor", supervisor_node)
        
        # Worker nodes
        self.graph.add_node("researcher", researcher_node)
        self.graph.add_node("analyst", analyst_node)
        self.graph.add_node("critic", critic_node)
        self.graph.add_node("writer", writer_node)
        
        # End node
        self.graph.add_node("end", end_node)
        
        # Error handler node
        self.graph.add_node("error", self._error_handler_node)
        
        logger.debug("Nodes added: supervisor, researcher, analyst, critic, writer, end, error")
        
        # =================================================================
        # Add Edges
        # =================================================================
        logger.debug("Adding edges...")
        
        # Entry point: START -> supervisor
        self.graph.add_edge(START, "supervisor")
        
        # Supervisor routes to workers (conditional edge)
        self.graph.add_conditional_edges(
            source="supervisor",
            path=route_from_supervisor,
            path_map={
                "researcher": "researcher",
                "analyst": "analyst",
                "critic": "critic",
                "writer": "writer",
                "end": "end",
                "error": "error",
            }
        )
        
        # Workers return to supervisor (conditional edges for error handling)
        self.graph.add_conditional_edges(
            source="researcher",
            path=route_after_researcher,
            path_map={
                "supervisor": "supervisor",
                "error": "error",
            }
        )
        
        self.graph.add_conditional_edges(
            source="analyst",
            path=route_after_analyst,
            path_map={
                "supervisor": "supervisor",
                "error": "error",
            }
        )
        
        self.graph.add_conditional_edges(
            source="critic",
            path=route_after_critic,
            path_map={
                "supervisor": "supervisor",
                "error": "error",
            }
        )
        
        self.graph.add_conditional_edges(
            source="writer",
            path=route_after_writer,
            path_map={
                "supervisor": "supervisor",
                "error": "error",
            }
        )
        
        # End and error nodes terminate workflow
        self.graph.add_edge("end", END)
        self.graph.add_edge("error", END)
        
        logger.debug("Edges configured")
        logger.info("Workflow graph built successfully")
        
        return self.graph
    
    def _error_handler_node(self, state: GraphState) -> GraphState:
        """
        Error handler node.
        
        Logs the error and marks workflow as failed.
        
        Args:
            state: Current graph state
            
        Returns:
            Updated state with failure status
        """
        logger.error("=" * 50)
        logger.error("ERROR HANDLER NODE")
        logger.error("=" * 50)
        
        error = state.get("error", "Unknown error")
        error_agent = state.get("error_agent", "Unknown")
        
        logger.error(f"Error from {error_agent}: {error}")
        
        return {
            **state,
            "workflow_status": "failed",
            "completed_at": datetime.now(),
        }
    
    def with_checkpointer(self, checkpointer: Optional[MemorySaver] = None) -> "WorkflowBuilder":
        """
        Add checkpointing for state persistence.
        
        Args:
            checkpointer: Custom checkpointer (default: MemorySaver)
            
        Returns:
            Self for chaining
        """
        self.checkpointer = checkpointer or MemorySaver()
        logger.info("Checkpointer configured")
        return self
    
    def compile(self) -> CompiledStateGraph:
        """
        Compile the workflow graph.
        
        Must be called after build().
        
        Returns:
            Compiled graph ready for execution
        """
        if self.graph is None:
            self.build()
        
        logger.info("Compiling workflow graph...")
        
        compile_args = {}
        if self.checkpointer:
            compile_args["checkpointer"] = self.checkpointer
        
        compiled = self.graph.compile(**compile_args)
        
        logger.info("Workflow graph compiled successfully")
        return compiled


# =============================================================================
# Workflow Runner
# =============================================================================

class WorkflowRunner:
    """
    Runner class for executing the multi-agent workflow.
    
    Provides methods for running queries through the workflow
    and handling results.
    """
    
    def __init__(
        self,
        api_key: str,
        max_iterations: int = 3,
        enable_checkpointing: bool = True,
    ):
        """
        Initialize the workflow runner.
        
        Args:
            api_key: Groq API key
            max_iterations: Maximum revision iterations
            enable_checkpointing: Whether to enable state checkpointing
        """
        self.api_key = api_key
        self.max_iterations = max_iterations
        
        # Build and compile workflow
        builder = WorkflowBuilder(api_key)
        
        if enable_checkpointing:
            builder.with_checkpointer()
        
        self.workflow = builder.build()
        self.compiled_workflow = builder.compile()
        self.checkpointer = builder.checkpointer
        
        logger.info(f"WorkflowRunner initialized (max_iterations={max_iterations})")
    
    def run(
        self,
        query: str,
        thread_id: Optional[str] = None,
        stream: bool = False,
    ) -> GraphState:
        """
        Run the workflow for a query.
        
        Args:
            query: User's research query
            thread_id: Optional thread ID for checkpointing
            stream: Whether to stream results (not yet implemented)
            
        Returns:
            Final graph state
        """
        logger.info("=" * 60)
        logger.info(f"STARTING WORKFLOW: {query[:50]}...")
        logger.info("=" * 60)
        
        # Create initial state
        initial_state = create_initial_state(
            user_query=query,
            max_iterations=self.max_iterations,
        )
        
        # Configure run
        config = {}
        if thread_id and self.checkpointer:
            config["configurable"] = {"thread_id": thread_id}
        
        try:
            # Run the workflow
            if stream:
                return self._run_streaming(initial_state, config)
            else:
                return self._run_sync(initial_state, config)
                
        except Exception as e:
            logger.error(f"Workflow execution error: {e}")
            return {
                **initial_state,
                "error": str(e),
                "workflow_status": "failed",
                "completed_at": datetime.now(),
            }
    
    def _run_sync(self, initial_state: GraphState, config: dict) -> GraphState:
        """
        Run workflow synchronously.
        
        Args:
            initial_state: Initial graph state
            config: Run configuration
            
        Returns:
            Final graph state
        """
        logger.info("Running workflow synchronously...")
        
        # Invoke the compiled workflow
        final_state = self.compiled_workflow.invoke(initial_state, config)
        
        # Log completion summary
        summary = get_state_summary(final_state)
        logger.info("=" * 60)
        logger.info("WORKFLOW COMPLETED")
        logger.info(f"Status: {summary['status']}")
        logger.info(f"Iterations: {summary['iteration']}")
        logger.info(f"Has Report: {summary['has_report']}")
        if summary['error']:
            logger.error(f"Error: {summary['error']}")
        logger.info("=" * 60)
        
        return final_state
    
    def _run_streaming(self, initial_state: GraphState, config: dict) -> GraphState:
        """
        Run workflow with streaming output.
        
        Args:
            initial_state: Initial graph state
            config: Run configuration
            
        Returns:
            Final graph state
        """
        logger.info("Running workflow with streaming...")
        
        final_state = initial_state
        
        # Stream through nodes
        for event in self.compiled_workflow.stream(initial_state, config):
            # Each event contains node outputs
            for node_name, node_output in event.items():
                logger.info(f"[STREAM] Node '{node_name}' completed")
                final_state = {**final_state, **node_output}
        
        return final_state
    
    def get_state(self, thread_id: str) -> Optional[GraphState]:
        """
        Get the current state for a thread.
        
        Requires checkpointing to be enabled.
        
        Args:
            thread_id: Thread identifier
            
        Returns:
            Current state or None
        """
        if not self.checkpointer:
            logger.warning("Checkpointing not enabled")
            return None
        
        config = {"configurable": {"thread_id": thread_id}}
        return self.compiled_workflow.get_state(config)


# =============================================================================
# Factory Functions
# =============================================================================

def create_workflow(api_key: str) -> CompiledStateGraph:
    """
    Create and compile the workflow graph.
    
    Simple factory function for quick workflow creation.
    
    Args:
        api_key: Groq API key
        
    Returns:
        Compiled workflow
    """
    builder = WorkflowBuilder(api_key)
    builder.build()
    return builder.compile()


def create_runner(
    api_key: str,
    max_iterations: int = 3,
    enable_checkpointing: bool = True,
) -> WorkflowRunner:
    """
    Create a workflow runner.
    
    Factory function for creating a configured runner.
    
    Args:
        api_key: Groq API key
        max_iterations: Maximum revision iterations
        enable_checkpointing: Enable state checkpointing
        
    Returns:
        Configured WorkflowRunner
    """
    return WorkflowRunner(
        api_key=api_key,
        max_iterations=max_iterations,
        enable_checkpointing=enable_checkpointing,
    )


# =============================================================================
# Visualization Helper
# =============================================================================

def get_workflow_diagram(api_key: str) -> str:
    """
    Generate a Mermaid diagram of the workflow.
    
    Useful for documentation and debugging.
    
    Args:
        api_key: Groq API key
        
    Returns:
        Mermaid diagram string
    """
    builder = WorkflowBuilder(api_key)
    builder.build()
    compiled = builder.compile()
    
    try:
        # LangGraph can generate Mermaid diagrams
        return compiled.get_graph().draw_mermaid()
    except Exception as e:
        logger.warning(f"Could not generate diagram: {e}")
        # Return a manual diagram
        return """
graph TD
    START([START]) --> supervisor
    supervisor --> |researcher| researcher
    supervisor --> |analyst| analyst
    supervisor --> |critic| critic
    supervisor --> |writer| writer
    supervisor --> |end| end_node
    supervisor --> |error| error
    researcher --> supervisor
    analyst --> supervisor
    critic --> supervisor
    writer --> supervisor
    end_node --> END([END])
    error --> END
"""


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Builder
    "WorkflowBuilder",
    # Runner
    "WorkflowRunner",
    # Factory functions
    "create_workflow",
    "create_runner",
    # Visualization
    "get_workflow_diagram",
]

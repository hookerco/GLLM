from typing import Annotated
from typing import TypedDict
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph

from gllm.utils.gcode_utils import generate_gcode_with_langchain, validate_syntax, validate_continuity, \
                              clean_gcode, validate_unreachable_code, validate_safety, validate_drilling_gcode, \
                              validate_functional_correctness

### Parameters
max_iterations = 50

class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        error : Binary flag for control flow to indicate whether test error was tripped
        messages : With user question, error messages, reasoning
        generation : Code solution
        iterations : Number of tries
    """

    error: str
    messages: Annotated[list[AnyMessage], add_messages]
    generation: str
    iterations: int

### Nodes
def generate(state: GraphState, chain, user_inputs):
    """
    Generate a code solution

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation
    """

    print("---GENERATING G-CODE SOLUTION---")

    # State
    messages = state["messages"]
    iterations = state["iterations"]
    error = state["error"]

    # Solution
    gcode_response = generate_gcode_with_langchain(chain, user_inputs)
    messages += [
        (
            "assistant",
            f"Here is my attempt to solve the problem: {user_inputs} \n Code: {gcode_response}",
        )
    ]

    # Increment
    iterations = iterations + 1
    return {"generation": gcode_response, "messages": messages, "iterations": iterations}

def code_check(state: GraphState, chain, user_inputs, parameters_string):
    """
    Check code

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, error
    """

    print("---CHECKING GENERATED G-CODE---")

    # State
    messages = state["messages"]
    code_solution = clean_gcode(state["generation"])
    iterations = state["iterations"]

    # Validate syntax
    is_valid_syntax, syntax_error_msg = validate_syntax(str(code_solution))
    if not is_valid_syntax:
        print("---Syntax CHECK: FAILED---")
        error_message = [("user", f"Your solution failed the Syntax test. Here is the error: {syntax_error_msg}. Reflect on this error and your prior attempt to solve the problem. (1) State what you think went wrong with the prior solution and (2) try to solve this problem again. Return the FULL SOLUTION.")]
        messages += error_message
        return {
            "generation": code_solution,
            "messages": messages,
            "iterations": iterations,
            "error": "yes",
        }
    
    # Check functional (semantic correctness)
    is_semantically_correct, semantic_error_msg = validate_functional_correctness(code_solution, parameters_string)
    if not is_semantically_correct and 'milling' in user_inputs['Operation Type']:
        print("---SEMANTIC CORRECTNESS CHECK: FAILED---")
        error_message = [("user", f"Your solution failed the code execution test: {semantic_error_msg}) Reflect on this error and your prior attempt to solve the problem. (1) State what you think went wrong with the prior solution and (2) try to solve this problem again. Return the FULL SOLUTION.")]
        messages += error_message
        return {
            "generation": code_solution,
            "messages": messages,
            "iterations": iterations,
            "error": "yes",
        }

    # Check continuty
    # is_continuous, continuity_error_msg = validate_continuity(code_solution)
    # if not is_continuous and 'milling' in user_inputs['Operation Type']:
    #     print("---CONTINUITY CHECK: FAILED---")
    #     error_message = [("user", f"Your solution failed the code execution test: {continuity_error_msg}) Reflect on this error and your prior attempt to solve the problem. (1) State what you think went wrong with the prior solution and (2) try to solve this problem again. Return the FULL SOLUTION.")]
    #     messages += error_message
    #     return {
    #         "generation": code_solution,
    #         "messages": messages,
    #         "iterations": iterations,
    #         "error": "yes",
    #     }

    # Check unreachable code
    is_unreachable_code, unreachable_error_msg = validate_unreachable_code(code_solution)
    if not is_unreachable_code:
        print("---UNREACHABLE CODE CHECK: FAILED---")
        error_message = [("user", f"Your solution failed the code execution test: {unreachable_error_msg}) Reflect on this error and your prior attempt to solve the problem. (1) State what you think went wrong with the prior solution and (2) try to solve this problem again. Return the FULL SOLUTION.")]
        messages += error_message
        return {
            "generation": code_solution,
            "messages": messages,
            "iterations": iterations,
            "error": "yes",
        }

    # Check safety
    is_safe_code, safety_error_msg = validate_safety(code_solution)
    if not is_safe_code:
        print("---SAFETY CHECK: FAILED---")
        error_message = [("user", f"Your solution failed the code execution test: {safety_error_msg}) Reflect on this error and your prior attempt to solve the problem. (1) State what you think went wrong with the prior solution and (2) try to solve this problem again. Return the FULL SOLUTION.")]
        messages += error_message
        return {
            "generation": code_solution,
            "messages": messages,
            "iterations": iterations,
            "error": "yes",
        }

    # Check correct drilling
    is_correct_drilling, drilling_error_msg = validate_drilling_gcode(code_solution)
    if not is_correct_drilling and 'drilling' in user_inputs['Operation Type']:
        print("---DRILLING CHECK: FAILED---")
        error_message = [("user", f"Your solution failed the code execution test: {drilling_error_msg}) Reflect on this error and your prior attempt to solve the problem. (1) State what you think went wrong with the prior solution and (2) try to solve this problem again. Return the FULL SOLUTION.")]
        messages += error_message
        return {
            "generation": code_solution,
            "messages": messages,
            "iterations": iterations,
            "error": "yes",
        }    
    
    
    # Check return to home position
    # is_return_to_home, return_error_msg = check_safe_return(code_solution)
    # if not is_return_to_home:
    #     print("---RETURN TO HOME CHECK: FAILED---")
    #     error_message = [("user", f"Your solution failed the code execution test: {return_error_msg}) Reflect on this error and your prior attempt to solve the problem. (1) State what you think went wrong with the prior solution and (2) try to solve this problem again. Return the FULL SOLUTION.")]
    #     messages += error_message
    #     return {
    #         "generation": code_solution,
    #         "messages": messages,
    #         "iterations": iterations,
    #         "error": "yes",
    #     }

    # No errors
    print("---NO G-CODE TEST FAILURES---")
    return {
        "generation": code_solution,
        "messages": messages,
        "iterations": iterations,
        "error": "no",
    }

### Conditional edges
def decide_to_finish(state: GraphState):
    """
    Determines whether to finish.

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """
    error = state["error"]
    iterations = state["iterations"]

    if error == "no" or iterations == max_iterations:
        print("---DECISION: FINISH---")
        return "end"
    else:
        print("---DECISION: RE-TRY SOLUTION---")
        return "generate"

def _print_event(event: dict, _printed: set, max_length=1500):
    current_state = event.get("dialog_state")
    if current_state:
        print(f"Currently in: ", current_state[-1])
    message = event.get("messages")
    if message:
        if isinstance(message, list):
            message = message[-1]
        if message.id not in _printed:
            msg_repr = message.pretty_repr(html=True)
            if len(msg_repr) > max_length:
                msg_repr = msg_repr[:max_length] + " ... (truncated)"
            print(msg_repr)
            _printed.add(message.id)

def construct_graph(model, user_inputs, parameters_string):
    builder = StateGraph(GraphState)

    # Define the nodes
    builder.add_node("generate", lambda state: generate(state, model, user_inputs))  # generation solution
    builder.add_node("check_code", lambda state: code_check(state, model, user_inputs, parameters_string))  # check code

    # Build graph
    builder.set_entry_point("generate")
    builder.add_edge("generate", "check_code")
    builder.add_conditional_edges(
        "check_code",
        decide_to_finish,
        {
            "end": END,
            "generate": "generate",
        },
    )

    memory = SqliteSaver.from_conn_string(":memory:")
    graph = builder.compile(checkpointer=memory)

    return graph

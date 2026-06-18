"""
Description of this file:

This is a Streamlit application that uses LLM pipelines with Langchain and Langgraph to generate G-codes for CNC machines. 
The application takes a natural language instruction as input and generates a G-code based on the instruction. 
The G-code is then validated and can be downloaded or visualized as a 3D plot.

The application is written in Python and uses the Streamlit library for the user interface. 
It also uses the Langchain and Langgraph libraries for the LLM pipelines.

Authors: Mohamed Abdelaal, Samuel Lokadjaja

This work was done at Software AG, Darmstadt, Germany in 2023-2024 and is published under the Apache License 2.0.
"""


import uuid
import streamlit as st
from gllm.proof.runner import ScenarioRequest, run_proof_scenario
from gllm.utils.model_utils import (
    DEFAULT_MODEL,
    MODEL_OPTIONS,
    get_openrouter_model_name,
    setup_model,
    setup_langchain_without_rag,
)
from gllm.utils.params_extraction_utils import extract_parameters_logic, display_extracted_parameters, parse_extracted_parameters, extract_numerical_values
from gllm.utils.gcode_utils import display_generated_gcode, generate_gcode_logic, plot_generated_gcode, validate_gcode, clean_gcode, generate_gcode_unstructured_prompt, generate_task_descriptions
from gllm.utils.graph_utils import construct_graph, _print_event
from gllm.utils.plot_utils import plot_user_specification, refine_gcode
from gllm.utils.params_extraction_utils import from_dict_to_text
    
DEFAULT_PROOF_REGISTRY = "config/vericut_setups.example.json"
DEFAULT_PROOF_OUTPUT_ROOT = ".proof-runs"


def extract_parameters(description_text):
        try:
            extracted_parameters, missing_parameters = extract_parameters_logic(st.session_state['langchain_chain'], description_text)
        except RuntimeError as exc:
            st.error(str(exc))
            st.stop()

        # update the relevant Streamlit states
        st.session_state['extracted_parameters'] = from_dict_to_text(extracted_parameters)
        st.session_state['missing_parameters'] = missing_parameters
        st.session_state['user_inputs'].update(extracted_parameters)


def get_or_setup_model(
    session_state,
    model_str,
    setup_model_fn=setup_model,
    openrouter_model_name=None,
):
    openrouter_selection = None
    if model_str == "OpenRouter" and openrouter_model_name:
        openrouter_selection = openrouter_model_name.strip() or None

    selection_key = (model_str, openrouter_selection)

    if session_state.get("selected_model") != selection_key:
        session_state["selected_model"] = selection_key
        session_state.pop("langchain_chain", None)
        session_state.pop("model_instance", None)

    if "model_instance" not in session_state:
        if model_str == "OpenRouter":
            session_state["model_instance"] = setup_model_fn(
                model_str,
                openrouter_model_name=openrouter_model_name,
            )
        else:
            session_state["model_instance"] = setup_model_fn(model_str)

    return session_state["model_instance"]


def run_existing_gcode_proof(
    *,
    prompt,
    gcode,
    registry_path,
    setup_id,
    output_root,
    run_vericut,
    model_name,
    timeout_seconds=None,
    scenario_id=None,
):
    request = ScenarioRequest(
        prompt=prompt,
        registry_path=registry_path,
        setup_id=setup_id,
        output_root=output_root,
        scenario_id=scenario_id,
        model_name=model_name,
        prompt_type="Streamlit generated candidate",
        run_vericut=run_vericut,
        max_repair_attempts=0,
        timeout_seconds=timeout_seconds,
    )
    return run_proof_scenario(
        request,
        candidate_generator=lambda _prompt, _context: gcode,
    )


def select_proof_candidate_gcode(candidate_gcode, generated_gcode):
    candidate = (candidate_gcode or "").strip()
    if candidate:
        return candidate
    return (generated_gcode or "").strip()


def proof_verdict_card(payload):
    status = payload.get("status", "unknown")
    operator_action = payload.get("operator_action", "manual_review_required")
    attempts = payload.get("final_attempt") or len(payload.get("attempts", []))
    severity = "info"
    if operator_action == "ready_to_review":
        severity = "success"
    elif operator_action in {"rerun_vericut", "manual_review_required"}:
        severity = "warning"
    elif operator_action in {"fix_prompt", "fix_setup", "reject"}:
        severity = "error"
    return {
        "headline": status,
        "severity": severity,
        "operator_action": operator_action,
        "attempts": attempts,
        "evidence_packet": payload.get("packet_file"),
    }


def display_proof_packet_summary(payload):
    card = proof_verdict_card(payload)
    message = (
        f"Proof status: {card['headline']} | "
        f"Operator action: {card['operator_action']}"
    )
    if card["severity"] == "success":
        st.success(message)
    elif card["severity"] == "warning":
        st.warning(message)
    elif card["severity"] == "error":
        st.error(message)
    else:
        st.info(message)

    columns = st.columns(3)
    columns[0].metric("Status", card["headline"])
    columns[1].metric("Action", card["operator_action"])
    columns[2].metric("Attempts", card["attempts"])
    if card["evidence_packet"]:
        st.write(f"Evidence packet: {card['evidence_packet']}")
    with st.expander("Evidence details"):
        st.json(payload)


def display_proof_run_controls(input_description, model_str):
    st.subheader("Proof Run")
    generated_gcode = st.session_state.get("gcode") or ""
    previous_generated_gcode = st.session_state.get("proof_candidate_generated_source", "")
    if "proof_candidate_gcode" not in st.session_state:
        st.session_state["proof_candidate_gcode"] = generated_gcode
    elif (
        generated_gcode
        and st.session_state["proof_candidate_gcode"] == previous_generated_gcode
    ):
        st.session_state["proof_candidate_gcode"] = generated_gcode
    st.session_state["proof_candidate_generated_source"] = generated_gcode

    candidate_gcode = st.text_area(
        "Candidate G-code",
        height=180,
        key="proof_candidate_gcode",
    )
    registry_path = st.text_input(
        "Vericut setup registry",
        value=DEFAULT_PROOF_REGISTRY,
    )
    setup_id = st.text_input(
        "Setup ID",
        value="vericut96_haas_minimill_sample",
    )
    output_root = st.text_input(
        "Proof output root",
        value=DEFAULT_PROOF_OUTPUT_ROOT,
    )
    run_vericut_checked = st.checkbox("Run Vericut batch simulation", value=False)
    timeout_seconds = st.number_input(
        "Vericut timeout seconds",
        min_value=1,
        value=900,
        step=30,
        disabled=not run_vericut_checked,
    )

    if st.button("Build proof packet"):
        proof_gcode = select_proof_candidate_gcode(candidate_gcode, generated_gcode)
        if not proof_gcode:
            st.error("Candidate G-code is required before building a proof packet.")
            return

        proof_prompt = input_description.strip() or "Manual Streamlit proof candidate"
        try:
            packet = run_existing_gcode_proof(
                prompt=proof_prompt,
                gcode=proof_gcode,
                registry_path=registry_path,
                setup_id=setup_id,
                output_root=output_root,
                run_vericut=run_vericut_checked,
                model_name=model_str,
                timeout_seconds=int(timeout_seconds) if run_vericut_checked else None,
            )
        except Exception as exc:
            st.error(f"Proof run failed: {exc}")
            return

        payload = packet.to_dict()
        st.session_state["proof_packet"] = payload
        display_proof_packet_summary(payload)


def main():

    _printed = set()
    thread_id = str(uuid.uuid4())
    config = {
        "configurable": {
            # Checkpoints are accessed by thread_id
            "thread_id": thread_id,},
            "recursion_limit": 1000}

    st.title("G-code Generator for CNC Machines")
    st.write("Please describe your CNC machining task in natural language:")
    input_description = st.text_area("Task Description", height=150)
    

    # Drop-down menu for model selection
    model_str = st.selectbox(
        'Choose a Language Model:',
        MODEL_OPTIONS,
        index=MODEL_OPTIONS.index(DEFAULT_MODEL),
    )
    openrouter_model_name = None
    if model_str == "OpenRouter":
        openrouter_model_name = st.text_input(
            "OpenRouter model ID",
            value=get_openrouter_model_name(),
        )

    try:
        model = get_or_setup_model(
            st.session_state,
            model_str,
            openrouter_model_name=openrouter_model_name,
        )
    except RuntimeError as exc:
        st.error(str(exc))
        st.stop()

    # Let the user choose whether to use structured or unstructured prompt
    prompt_type = st.selectbox('Prompt Type:', ('Structured', 'Unstructured'), index=0)
    
    pdf_files = st.file_uploader("Upload PDF files with additional knowledge (RAG)", accept_multiple_files=True, type=['pdf'])

    if "langchain_chain" not in st.session_state:
        try:
            if pdf_files:
                from gllm.utils.rag_utils import setup_langchain_with_rag

                st.session_state['langchain_chain'] = setup_langchain_with_rag(pdf_files, model)
            else:
                st.session_state['langchain_chain'] = setup_langchain_without_rag(model=model)
        except RuntimeError as exc:
            st.error(str(exc))
            st.stop()
        
    if "extracted_parameters" not in st.session_state:
        st.session_state['extracted_parameters'] = None
        st.session_state['missing_parameters'] = None
        st.session_state['user_inputs'] = {}
        st.session_state['gcode'] = None
        st.session_state['task_descriptions'] = []
        st.session_state['decompose_task'] = None
        st.session_state['extracted_parameters_backup'] = None
        st.session_state['user_inputs_backup'] = {}
        
    if "parsed_parameters" not in st.session_state:
        st.session_state.parsed_parameters = {} 

    #################################################
    ############# Parameters Extraction #############
    #################################################

 

    disable_extract_button = False if prompt_type == 'Structured' else True    # Disable Parameter Extraction if user selects unstructured prompt 

    # user selects whether to use the task decomposor 
    st.session_state['decompose_task'] = st.selectbox("Decompose The task Description: ", ('Yes', 'No'), index=0, disabled=disable_extract_button)  

    extract_button = st.button("Extract Parameters", disabled=disable_extract_button)   
    if extract_button and "langchain_chain" in st.session_state:
        extract_parameters(description_text=input_description)
        st.session_state['extracted_parameters_backup'] = st.session_state['extracted_parameters']
        st.session_state['user_inputs_backup'] = st.session_state['user_inputs']

        # generate subtask descriptions if the input task invovles more than one shape
        values_in_number_shapes = extract_numerical_values(st.session_state['user_inputs'], 'Number of Shapes')
        number_shapes = values_in_number_shapes[0] if type(values_in_number_shapes) is list else values_in_number_shapes
        
        if number_shapes > 1 and st.session_state['decompose_task'] == 'Yes':
            st.session_state['task_descriptions'] = generate_task_descriptions(model, model_str, input_description)
            st.session_state['extracted_parameters'] += f"Subtasks: {st.session_state['task_descriptions']}\n"
        else:
            st.session_state['task_descriptions'] = [input_description]
        
    if st.session_state['extracted_parameters']:
        display_extracted_parameters()

    if st.button("Simulate the tool path (2D)", disabled=disable_extract_button):
        if st.session_state['extracted_parameters']:
            st.session_state['parsed_parameters'] = parse_extracted_parameters(st.session_state['extracted_parameters'])
            st.text("If the plotted path is incorrect, please adjust the task description.")
            st.pyplot(plot_user_specification(parsed_parameters=st.session_state.parsed_parameters)) 

    ################################################
    ############ G-Code Generation #################
    ################################################

    if st.button("Generate G-code"):

        gcodes_combined = ""
        
        if not st.session_state['task_descriptions']:
            st.session_state['task_descriptions'] = [input_description]

        for subtask_description in st.session_state['task_descriptions']:
            print("++++++++++++++++++++++++++++++++++++++++++")
            print("SUBTASK DESCRIPTION:", subtask_description)
            print("++++++++++++++++++++++++++++++++++++++++++")
            if disable_extract_button:
                st.session_state['gcode'] = generate_gcode_unstructured_prompt(st.session_state['langchain_chain'], subtask_description)
            else:
                
                if len(st.session_state['task_descriptions']) > 1:
                    extract_parameters(description_text=subtask_description)

                if "langchain_chain" in st.session_state and 'parsed_parameters' in st.session_state:
                    # construct graph
                    graph = construct_graph(st.session_state['langchain_chain'], st.session_state['user_inputs'], st.session_state['extracted_parameters'])
                    events = graph.stream({"messages": [("user", subtask_description)], "iterations": 0}, config, stream_mode="values")
                    for event in events:
                       pass 
                        #_print_event(event, _printed)

                    gcodes_combined += f"\n{event['generation']}"
                    gcodes_combined = refine_gcode(gcodes_combined) 
        
                    st.session_state['gcode'] = gcodes_combined

        # restore the extracted parameters from the input task description
        st.session_state['user_inputs'] = st.session_state['user_inputs_backup']
        st.session_state['extracted_parameters'] = st.session_state['extracted_parameters_backup']

    display_generated_gcode()

    plot_generated_gcode()

    display_proof_run_controls(input_description, model_str)

     # Debug information
    if st.checkbox("Show Debug Info"):
        st.write("Session State:", st.session_state)

if __name__ == "__main__":
    main()

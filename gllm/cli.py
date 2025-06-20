import argparse
import uuid

from gllm.utils.rag_utils import setup_langchain_with_rag
from gllm.utils.model_utils import setup_model, setup_langchain_without_rag
from gllm.utils.params_extraction_utils import (
    extract_parameters_logic,
    parse_extracted_parameters,
    extract_numerical_values,
    from_dict_to_text,
)
from gllm.utils.gcode_utils import (
    generate_gcode_unstructured_prompt,
    generate_task_descriptions,
)
from gllm.utils.graph_utils import construct_graph
from gllm.utils.plot_utils import plot_user_specification, plot_gcode, refine_gcode


def ask_yes_no(question: str) -> bool:
    """Simple helper to ask yes/no questions on the CLI."""
    answer = input(f"{question} [y/N]: ").strip().lower()
    return answer in {"y", "yes"}


def load_pdf_files(paths):
    files = []
    for p in paths:
        try:
            files.append(open(p, "rb"))
        except OSError as e:
            print(f"Could not open {p}: {e}")
    return files


def main():
    parser = argparse.ArgumentParser(description="G-code Generator CLI")
    parser.add_argument(
        "--model",
        choices=["Zephyr-7b", "GPT-3.5", "Fine-tuned StarCoder", "CodeLlama", "OpenRouter"],
        default="GPT-3.5",
        help="Language model to use",
    )
    parser.add_argument(
        "--prompt-type",
        choices=["Structured", "Unstructured"],
        default="Structured",
        help="Whether to use the structured or unstructured prompt",
    )
    parser.add_argument(
        "--pdfs",
        nargs="*",
        default=None,
        help="Optional PDF files to use for RAG",
    )
    parser.add_argument(
        "--decompose-task",
        choices=["Yes", "No"],
        default="Yes",
        help="Decompose the task description when multiple shapes are detected",
    )
    parser.add_argument(
        "--output",
        help="Optional file to save the generated G-code",
    )
    args = parser.parse_args()

    model = setup_model(model=args.model)

    if args.pdfs:
        pdf_files = load_pdf_files(args.pdfs)
        chain = setup_langchain_with_rag(pdf_files, model)
    else:
        chain = setup_langchain_without_rag(model=model)

    description = input("Please describe your CNC machining task:\n")

    user_inputs = {}
    task_descriptions = []

    if args.prompt_type == "Structured":
        extracted_parameters, missing_parameters = extract_parameters_logic(
            chain, description
        )
        print("\nExtracted Parameters:")
        for k, v in extracted_parameters.items():
            print(f"- {k}: {v}")

        for param in list(missing_parameters):
            value = input(f"Please provide the {param}: ")
            if value:
                extracted_parameters[param] = value
                user_inputs[param] = value

        user_inputs.update(extracted_parameters)

        values = extract_numerical_values(user_inputs, "Number of Shapes")
        number_shapes = values[0] if isinstance(values, list) else values
        if number_shapes > 1 and args.decompose_task == "Yes":
            task_descriptions = generate_task_descriptions(model, args.model, description)
            print("\nGenerated subtask descriptions:")
            for t in task_descriptions:
                print("-", t)
        else:
            task_descriptions = [description]

        if ask_yes_no("Simulate the tool path (2D)?"):
            parsed = parse_extracted_parameters(from_dict_to_text(user_inputs))
            plot_module = plot_user_specification(parsed)
            plot_module.show()
    else:
        task_descriptions = [description]

    if ask_yes_no("Generate G-code?"):
        gcode_combined = ""
        for subtask in task_descriptions:
            if args.prompt_type == "Unstructured":
                gcode_combined += "\n" + generate_gcode_unstructured_prompt(chain, subtask)
            else:
                if len(task_descriptions) > 1:
                    params, _ = extract_parameters_logic(chain, subtask)
                    user_inputs.update(params)
                    parameters_string = from_dict_to_text(params)
                else:
                    parameters_string = from_dict_to_text(user_inputs)
                graph = construct_graph(chain, user_inputs, parameters_string)
                thread_id = str(uuid.uuid4())
                config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 1000}
                events = graph.stream({"messages": [("user", subtask)], "iterations": 0}, config, stream_mode="values")
                for event in events:
                    pass
                gcode_combined += f"\n{event['generation']}"
                gcode_combined = refine_gcode(gcode_combined)

        gcode = gcode_combined.strip()
        print("\nGenerated G-code:\n")
        print(gcode)

        if args.output:
            with open(args.output, "w") as f:
                f.write(gcode)
            print(f"G-code saved to {args.output}")

        if ask_yes_no("Plot G-code?"):
            plot_module = plot_gcode(gcode)
            plot_module.show()

    if ask_yes_no("Show debug info?"):
        debug = {
            "model": args.model,
            "prompt_type": args.prompt_type,
            "user_inputs": user_inputs,
            "task_descriptions": task_descriptions,
        }
        print(debug)


if __name__ == "__main__":
    main()

from transformers import AutoModelWithLMHead, AutoTokenizer

# Load the original model and tokenizer
model = AutoModelWithLMHead.from_pretrained("bigcode/starcoderbase-3b")
tokenizer = AutoTokenizer.from_pretrained("bigcode/starcoderbase-3b")

# If the tokenizer does not have a padding token, set it to be the same as eos_token
if tokenizer.pad_token is None: 
    tokenizer.pad_token = tokenizer.eos_token

# Load the adapter
model.load_adapter("ArneKreuz/starcoderbase-finetuned-thestack")

# Set the model to use the adapter
model.active_adapters = "starcoderbase-finetuned-thestack"

instructions = """Generate G-code with the following instructions: 
Machine: Siemens 840D CNC Mill
Operation: Milling
Material: Steel
Workpiece Dimensions: 100mm x 100mm x 20mm
Tool Selection:
  - Milling Tool: Carbide end mill, 10mm diameter, 4-flute
Toolpath Strategy:
  - Pocket milling to remove material and create desired features
  - Contour milling for finishing passes
Machining Parameters:
  - Pocket Milling:
    - Cutting Speed: 500 m/min
    - Feed Rate: 200 mm/min
    - Depth of Cut: 5 mm
    - Stepover: 50%
  - Contour Milling:
    - Cutting Speed: 800 m/min
    - Feed Rate: 300 mm/min
    - Stepdown: 1 mm
Safety Considerations:
  - Rapid Retract Height: 10 mm
  - Tool Clearance: 2 mm

"""

# Encode the instructions and generate the G-code
input_ids = tokenizer.encode(instructions, return_tensors='pt')
attention_mask = input_ids.ne(tokenizer.pad_token_id).int()
output = model.generate(input_ids, attention_mask=attention_mask, max_length=1000, temperature=0.7)

# Decode the output to get the G-code
g_code = tokenizer.decode(output[0])

with open('output.txt', 'w') as f:
    f.write(g_code)
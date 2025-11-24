from llama_cpp import Llama
from huggingface_hub import hf_hub_download
from lmformatenforcer import JsonSchemaParser
from lmformatenforcer.integrations.llamacpp import build_llamacpp_logits_processor
from llama_cpp import LogitsProcessorList
from pydantic import BaseModel

def generate_structured_data(input_data: str, pydantic_class: BaseModel, model_path: str = "TheBloke/Llama-2-7b-Chat-GGUF") -> dict:
    """
    Generate structured data according to a Pydantic class.

    Args:
    - input_data (str): The input data to be processed.
    - pydantic_class (BaseModel): The Pydantic class to structure the data.
    - model_path (str): The path to the Llama model.

    Returns:
    - dict: The structured data according to the Pydantic class.
    """

    # Download the Llama model if it doesn't exist
    downloaded_model_path = hf_hub_download(repo_id=model_path, filename="llama-2-7b-chat.Q5_K_M.gguf")
    
    # Initialize the Llama model
    llm = Llama(model_path=downloaded_model_path)

    # Define the system prompt
    DEFAULT_SYSTEM_PROMPT = """\
    You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\
    """

    # Define the function to get the prompt
    def get_prompt(message: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> str:
        return f'<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{message} [/INST]'

    # Define the function to generate structured data
    def llamacpp_with_json_schema_parser(llm: Llama, prompt: str, json_schema_parser: JsonSchemaParser) -> str:
        logits_processors: LogitsProcessorList = LogitsProcessorList([build_llamacpp_logits_processor(llm, json_schema_parser)])
        output = llm(prompt, logits_processor=logits_processors)
        text: str = output['choices'][0]['text']
        return text

    # Define the question and prompt
    question = f'Please generate structured data according to the following JSON schema: {pydantic_class.schema_json()}. The input data is: {input_data}'
    prompt = get_prompt(question)

    # Generate the structured data
    json_schema_parser = JsonSchemaParser(pydantic_class.schema())
    result = llamacpp_with_json_schema_parser(llm, prompt, json_schema_parser)

    # Return the structured data as a dictionary
    return eval(result)

# Example usage:
class PlanetSchema(BaseModel):
    planet_name: str

class PlanetList(BaseModel):
    planets: list[PlanetSchema]

input_data = "Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, Neptune"
result = generate_structured_data(input_data, PlanetList)
print(result)

# Standard library imports
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:21'

# Third-party library imports
from dotenv import main
from fastapi import FastAPI, Request, HTTPException, status
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address
import torch
torch.cuda.empty_cache()
from torch import cuda, bfloat16
import transformers
import uvicorn
import pinecone

# Local (application-specific) imports
from langchain.chains import RetrievalQA
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.vectorstores import Pinecone
from langchain.schema import BaseRetriever
from langchain.agents import initialize_agent, load_tools, AgentOutputParser
from langchain.schema import AgentAction, AgentFinish
from langchain.memory import ConversationBufferWindowMemory, ConversationBufferMemory
from langchain import LLMChain, PromptTemplate


# Load environment variables
main.load_dotenv()

# Setup device configuration
device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

# Initialize embedding model
embed_model_id = 'sentence-transformers/all-MiniLM-L6-v2'
embed_model = HuggingFaceEmbeddings(
    model_name=embed_model_id,
    model_kwargs={'device': device},
    encode_kwargs={'device': device, 'batch_size': 32}
)

# Initialize Pinecone vectorstore DB
pinecone.init(
    api_key=os.environ['PINECONE_API_KEY'], 
    environment=os.environ['PINECONE_ENVIRONMENT']
)
pinecone.whoami()
index_name = 'llama'
index = pinecone.Index(index_name)
index.describe_index_stats()

# Set quantization configuration to load large model with less GPU memory
bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=bfloat16
)

# Initialize HuggingFace pipeline with the required model
hf_auth = os.environ['HUGGING_FACE_TOKEN']
model_id = 'meta-llama/Llama-2-13b-chat-hf'
#model_id = 'meta-llama/Llama-2-7b-chat-hf'
model_config = transformers.AutoConfig.from_pretrained(
    model_id,
    use_auth_token=hf_auth
)
model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    config=model_config,
    quantization_config=bnb_config,
    device_map='auto',
    use_auth_token=hf_auth
)
model.eval()
print(f"Model loaded on {device}")

# Initialize tokenizer
tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_id,
    use_auth_token=hf_auth
)

# Initialize text generation pipeline
generate_text = transformers.pipeline(
    model=model, tokenizer=tokenizer,
    return_full_text=True,  # langchain expects the full text
    task='text-generation',
    temperature=0.0,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
    max_new_tokens=512,  # max number of tokens to generate in the output
    repetition_penalty=1.1  # without this output begins repeating
)

# Create HuggingFacePipeline
llm = HuggingFacePipeline(pipeline=generate_text)

# Initiate pinecone Retrieval QA chain
text_field = 'text' 
vectorstore = Pinecone(
            index, embed_model.embed_query, text_field
)


# Create stop list and transform it to token ids
stop_list = ['\nHuman:', '\n```\n']
stop_token_ids = [tokenizer(x)['input_ids'] for x in stop_list]
stop_token_ids = [torch.LongTensor(x).to(device) for x in stop_token_ids]

# Define custom stopping criteria object
class StopOnTokens(transformers.StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stop_ids in stop_token_ids:
            if torch.eq(input_ids[0][-len(stop_ids):], stop_ids).all():
                return True
        return False

# Create the stopping criteria list
stopping_criteria = transformers.StoppingCriteriaList([StopOnTokens()])

# Create text generation pipeline
generate_text = transformers.pipeline(
    model=model, tokenizer=tokenizer,
    return_full_text=True,
    task='text-generation',
    temperature=0.0,
    max_new_tokens=512,
    repetition_penalty=1.1
)

# Load tools
llm = HuggingFacePipeline(pipeline=generate_text)

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""


def get_prompt(instruction, new_system_prompt=DEFAULT_SYSTEM_PROMPT ):
    SYSTEM_PROMPT = B_SYS + new_system_prompt + E_SYS
    prompt_template =  B_INST + SYSTEM_PROMPT + instruction + E_INST
    return prompt_template

def cut_off_text(text, prompt):
    cutoff_phrase = prompt
    index = text.find(cutoff_phrase)
    if index != -1:
        return text[:index]
    else:
        return text

def remove_substring(string, substring):
    return string.replace(substring, "")



def generate(text):
    prompt = get_prompt(text)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        inputs = tokenizer(prompt, return_tensors="pt").to('cuda')
        outputs = model.generate(**inputs,
                                 max_new_tokens=512,
                                 eos_token_id=tokenizer.eos_token_id,
                                 pad_token_id=tokenizer.eos_token_id,
                                 )
        final_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        final_outputs = cut_off_text(final_outputs, '</s>')
        final_outputs = remove_substring(final_outputs, prompt)

    return final_outputs#, outputs

def parse_text(text):
        wrapped_text = textwrap.fill(text, width=100)
        print(wrapped_text +'\n\n')
        # return assistant_text

instruction = "Chat History:\n\n{chat_history} \n\nUser: {user_input}"
system_prompt = "You are a helpful assistant, you always only answer for the assistant then you stop. Read the chat history to get context"

template = get_prompt(instruction, system_prompt)
print(template)

prompt = PromptTemplate(
    input_variables=["chat_history", "user_input"], template=template
)
memory = ConversationBufferMemory(memory_key="chat_history")

llm_chain = LLMChain(
    llm=llm,
    prompt=prompt,
    verbose=True,
    memory=memory,
)

######### Server ##########

class Query(BaseModel):
    user_input: str
    user_id: str

def get_user_id(request: Request):
    try:
        body = parse_obj_as(Query, request.json())
        user_id = body.user_id
        return user_id
    except Exception as e:
        return get_remote_address(request)


# Define FastAPI app
app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="./static/BBALP00A.TTF")

# Define limiter
limiter = Limiter(key_func=get_user_id)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

@app.exception_handler(RateLimitExceeded)
async def custom_rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        content={"detail": "Too many requests, please try again in an hour."},
    )


# Initialize user state
user_states = {}

# Define FastAPI endpoints

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/_health")
async def health_check():
    return {"status": "OK"}


@app.post('/gpt')
@limiter.limit("10/hour")
def react_description(query: Query, request: Request):
    print(query.user_input)
    user_id = query.user_id
    if user_id not in user_states:
        user_states[user_id] = None

    try:
        
        print('checkpoint_1')
        user_query = query.user_input
        retriever = vectorstore.similarity_search(
            user_query,  # the search query
            k=1 # returns top 3 most relevant chunks of text
        )
        document = retriever[0]  # Get the first document
        page_content = document.page_content
        print(page_content) 
        
        result = llm_chain.predict(user_input=user_query) 
        print(result)
              
        return {'response': result}

        
    except ValueError as e:
        print(e)
        raise HTTPException(status_code=400, detail="Invalid input")


# uvicorn simple_chat:app --reload --port 8008

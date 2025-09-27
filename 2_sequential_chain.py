from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

#Every time u create a new workflow(project) it requires a separate langsmith project which is mentioned in the .env . we can either change the langsmith project name in .env or u can create a new langsmith project inside .py file where the workflow is present .
import os 
os.environ['LANGCHAIN_PROJECT'] = 'Sequential LLM App'
load_dotenv()

prompt1 = PromptTemplate(
    template='Generate a detailed report on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Generate a 5 pointer summary from the following text \n {text}',
    input_variables=['text']
)

model1 = ChatOpenAI(model = 'gpt-4o-mini',temperature=0.7)

model2 = ChatOpenAI(model = 'gpt-4o-mini',temperature=0.5)

parser = StrOutputParser()

chain = prompt1 | model1 | parser | prompt2 | model2 | parser
#to set ur own metadata and tags in trace in langsmith . just for nice understanding in the langsmith webiste  (present in langsmith website - > tracing project)
config ={
    'run_name':'sequential_chain', #to change the name of trace 
    'tags':['llm app','report generation','summarization'],
    'metadata':{'model':'gpt-40-mini','model1_temp':0.7,'parser':'stroutparser'}
}

result = chain.invoke({'topic': 'Unemployment in India'},config = config)

print(result)

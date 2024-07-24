from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.weaviate import Weaviate
from langchain.document_loaders.text import TextLoader
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models.openai import ChatOpenAI
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_openai.embeddings import OpenAIEmbeddings
from fastapi import FastAPI 
import weaviate
from dotenv import load_dotenv
load_dotenv()
app = FastAPI()

import os





async def setup_chain(max_tokens , redo , query):
    # Load the json data to LLM 
    
    loader = TextLoader(
        file_path = './data/algonorm.txt' ,
        autodetect_encoding= True
    ) 
    data = loader.load()
    # print(data)
    # Chunking documents
    
    text_splitter = CharacterTextSplitter(chunk_size=1000 , chunk_overlap=50 )
    chunks = text_splitter.split_documents(data)
    #  Loading these chunks to vector database




    client = weaviate.Client(
        url=os.getenv("URL"),
        auth_client_secret=weaviate.auth.AuthApiKey(os.getenv("AUTH")), 
        additional_headers={ 
            "X-OpenAI-Api-Key":os.getenv("OPEN_AI_KEY") ,            # Replace with your OpenAI key
        }     
    )
    
# Assuming 'chunks' is a list of dictionaries representing your data
# Create a Weaviate vector store from the documents
    print("chunks---------------" , chunks)
    vectorstore = Weaviate.from_documents(
        client= client,
        documents = chunks,
        embedding=OpenAIEmbeddings(api_key=os.getenv("OPEN_AI_KEY")),
        by_text=False
    )
    
    

    
    retriever = vectorstore.as_retriever()
    print("retriever----------------" , retriever)
    
    
    # chunks--------------- [Document(page_content="https://algonomy.com\nUncover 2024's HOTTEST ecommerce personalization trends & master customer engagement. Explore Now!\nCommended by top Industry Analysts as a Leading Innovator in Retail\nTrusted by Digital\nteams across 400+ Retail and eCommerce Companies\nOmnichannel Customer Marketing\nImprove Customer Loyalty and Lifetime Value\nDrive in-the-moment engagement, on your customers’ preferred channels, with highly contextual and hyper-personalized campaigns. Leverage constantly competing, behavioral algorithms that optimize segmentation strategies in real time.\nCampaign audiences automatically built by combining marketer goals with customer behavior, affinity, and preferences using advanced algorithms.\nAI that orchestrates customer journeys and activation across touchpoints with continuous experimentation and testing.\nA content hub that dynamically brings together data across sources to generate holistic, contextual visual content personalized for each customer.\nA data platform where 360-degree customer profiles are continuously built, augmented, enriched, and made available in real time.\nLearn more\nDigital Experience Personalization\nBoost Engagement, Conversion Rates, and Order Values\nDeliver highly personalized experiences, powered by a patented, industry leading AI decisioning platform.\nPersonalize the end-to-end customer experience, from search or browse, extending through the cart and checkout phases, all the way to post-purchase service.\nDeliver fully customized and guided experiences that go beyond individual product recommendations, to fully personalized outfits, bundles, social proof, and other interactive experiences.\nLeverage a unique personalized search platform tailored to B2C and B2B commerce use cases.\nEnjoy automated catalog enrichment, powered by ChatGPT and LLM algorithms, as a significant value add to merchandisers, marketers, and analytics.\nLearn more\nMerchandising and Supply Chain Optimization\nTake the guesswork out of your merchandising and supply chain decisions\nElevate shopper experiences, step up your inventory game, achieve profitable growth and accelerate time-to-insight with AI-powered hyperlocal optimizations, seamless automation, groundbreaking collaboration and ready-to-roll analytics.\nBest-in-class demand forecasting AI platform, to power predictive decisions across the retail category planning and management\nSpecialized, advanced analytics and algorithms that optimize assortment planning, buying and allocation, promotions, and markdowns.\nBuilt-for-retail AI that powers inventory and replenishment decisions, designed to optimize inventory productivity and enhance availability of SKUs across stores, locations and channels.\nUnique supplier collaboration capabilities for integrated category and supply chain management with your brand and supply partners.\nBusiness Intelligence platform built for retail scale with role-based analytics for category managers, buyers, planners, and site (eCommerce) merchants.\nLearn more\nEnterprise Scale. Algorithm-first Approach. Unparalleled Retail Expertise.\n400+\nEnterprise Brands\n20\nCountries\n22\nTechnology Patents\n1.2B+\nCustomer Events Processed Every Day\n30B+\nAlgorithmic Decisions Made Every Day\nEnjoy Rapid Time to Market\nCustomize your tech stack with our ecosystem of 560+ connectors across categories, including eCommerce and Marketplaces, CRM, Marketing Automation, Analytics, Social Media, and more.\nAcknowledged by Industry Experts for 20+ years of Ground Breaking Retail Technology Innovations\nOur Clients Love Us Here’s what they have to say\n“The CDP and Customer Journey Orchestration projects were key to McDonald’s digital and data transformation journey, whereby we were able to build capabilities to drive insights-driven marketing across channels. We deeply appreciate the invaluable assistance provided by Algonomy’s analytics and campaign specialists, who have worked closely with us to develop and optimize our campaigns.”\nArvind RP\nCMO, McDonald’s\n+40%\nIncrease in omnichannel customers\n33%\nYoY increase in customers using McDelivery Services\n44 million\nCustomer engagement opportunities created across 6 channels\nRead Case Study\nBLOG\nAI-Driven Replenishment Planning: A Game-Changer for Retailers\nRead More\nGUIDE\nThe Ultimate Guide to Demand Forecasting in Grocery Retail (2023)\nRead More\nPrevious\nNext\nAlgonomy (previously Manthan-RichRelevance) empowers leading brands to become digital-first with the industry’s only real-time Algorithmic Decisioning platform that unifies data, decisioning, and orchestration. With industry-leading retail AI expertise connecting demand to supply with a real-time customer data platform as the foundation, Algonomy enables 1:1 omnichannel personalization, customer journey orchestration & analytics, merchandising analytics, and supplier collaboration.\nPRODUCT\nCOMPANY \nWe use cookies on our website to give you the most relevant experience by remembering your preferences and repeat visits. By clicking “Accept”, you consent to the use of ALL the cookies. Read More\nCookie Settings\nAccept", metadata={'source': './data/algonorm.txt'})]
    # #  Augmenting the output
    
    
    template = """
        ###Instructions###
        ROLE : You are a Chat Bot Assistant bot to help with queries asked by customers.
        IMPORTANT : Context is your only knowledge base.Don’t answer, if the information asked or relevant information in query is not mentioned in the Context. Answer only if the information is present in provided context.Recognize the keywords in Query and provide only the asked information if present in the Context. Provide reference website links only present in Context.Keep the tone of the responses as formal.When using Technical Jargons, try to provide a short explanation for it as well in order to maintain ease of understanding.
        If information is not present in Context just provide an apologise for not providing an answer in a format "I'm sorry I could not find a suitable response. If you have any other questions or need assistance with anything else, please feel free to ask.". DO NOT provide related information. 
        Follow best practices that results in structured response for clarity. For reference, you can rely on these guidelines -
        -Use bold formatting for headings to highlight key sections.
        -Describe the summary utilizing bullet points.
        -Utilize bullet points to list essential pieces of information.
        -If you provide any links, make sure they are clickable.
        
        ###Examples to understand only and not use as context
            Context : "Docker is a containerization software and helps streamline devops process. Reference [http://www.docker.com]"
            Query : "What is devops and provide links for understanding how devops work ?"
            Answer : "Information not found in knowledge-base" 
            
            Context : "Chocolate Cake is made up of flour , chocolate , eggs , butter and sugar and yeast. They are easy to make and are delicious to eat. Reference [http://www.Cakes.com]"
            Query : "What are Chocolate Cakes and provide link for references ?"
            Answer : "Chocolate Cakes are made up of flour , eggs , butter and sugar and are easy to make and are delicious to eat. Reference [http://www.Cakes.com]" 
        ###
        
        
        ###Information###
        Context : {context}
        Question : {question}
        
    """
    
    template2 = """
            ###Instructions
            ROLE : You are a customer support Assistant bot to help with queries asked by customers.
            IMPORTANT : Context is your only knowledge base.Don’t answer, if the information asked or relevant information in query is not mentioned in the Context. Answer only if the information is present in provided context.Recognize the keywords in Query and provide only the asked information if present in the Context. Provide reference website links only present in Context.Keep the tone of the responses as formal.When using Technical Jargons, try to provide a short explanation for it as well in order to maintain ease of understanding.
            If information is not present in Context just apologise for the inconveniece.DO NOT provide related information. 
            Follow best practices that results in structured response for clarity. For reference, you can rely on these guidelines -
            -Use bold formatting for headings to highlight key sections.
            -Describe the summary utilizing bullet points.
            -Utilize bullet points to list essential pieces of information.
            -If you provide any links, make sure they are clickable.
            ###
            
            ###Examples to understand only and not use as context
                Context : "Docker is a containerization software and helps streamline devops process. Reference [http://www.docker.com]"
                Query : "What is devops and provide links for understanding how devops work ?"
                Answer : "Information not found in knowledge-base" 
                
                Context : "Chocolate Cake is made up of flour , chocolate , eggs , butter and sugar and yeast. They are easy to make and are delicious to eat. Reference [http://www.Cakes.com]"
                Query : "What are Chocolate Cakes and provide link for references ?"
                Answer : "Chocolate Cakes are made up of flour , eggs , butter and sugar and are easy to make and are delicious to eat. Reference [http://www.Cakes.com]" 
            ###
        
        The answer generated for this question  using above Instruction Template was not liked by the user. Please re-generate this answer from the provided Context with more relevant information and better structure and format.
        Question: {question}
        Context : {context}
    """
    prompt1 = ChatPromptTemplate.from_template(template)
    prompt2 =  ChatPromptTemplate.from_template(template2)
    
    if(redo):
        prompt = prompt2
    else:
        prompt = prompt1
    
    
    
    # # # generating output
    
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=1 , max_tokens=max_tokens , openai_api_key=os.getenv("OPEN_AI_KEY"))
    

    rag_chain = (
        {"context": retriever,  "question": RunnablePassthrough()} 
        | prompt
        | llm
        | StrOutputParser()   
    )

    return rag_chain.invoke(query)
    
    
    



@app.get('/')
async def function():
    response  = await setup_chain(1000 , False ,  "what are subatomic particles?")
    print(response)
    return "Backend Port is working"




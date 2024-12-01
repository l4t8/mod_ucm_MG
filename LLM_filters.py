import os # Interactuar con archivos
import openai # Utilizar la API de openai
from openai import OpenAI 
from langchain_community.document_loaders.csv_loader import CSVLoader
# Cargar un documento CSV en langchain
from langchain.embeddings.openai import OpenAIEmbeddings
# Crear embeddings
from langchain_chroma import Chroma
# Importa Chroma, un vector database muy ligero
from dotenv import load_dotenv

# Cargar variables de entorno desde el archivo .env
load_dotenv()

# Acceder a la clave API
os.environ["OPENAI_API_KEY"] = openai.api_key = os.getenv("OPENAI_API_KEY")
# Verificar que la clave está cargada (opcional)
if not openai.api_key:
    raise ValueError("La clave de openai API no funciona o no existe.")


# Se utiliza un modelo de "embeddings" de openai que consuma pocos recursos.
embedding = OpenAIEmbeddings(model="text-embedding-ada-002")

vector_db_directory = 'files/chroma'
def generate_vectorstore(path_to_csv):
    """
    Inicializa el Job Loader, que contiene archivos del tipo 
    'langchain_core.documents.base.Document', documentos que pueden 
    ser interpretados por langchain.

    Posteriormente, genera un 'vectorshore of embeddings'. Este proceso 
    es costoso en comparación con lo que suele costar una llamada a la 
    API y solo debe ejecutarse cuando se modifique el 'dataset' original.
    """

    loader = CSVLoader(file_path=path_to_csv)
    JL = loader.load()

    return Chroma.from_documents(documents=JL,
                                 embedding=embedding,
                                 persist_directory=vector_db_directory)
        

def load_vectorstore():
    """
    Carga un 'vectorshore of embeddings'. Solo funciona si se ha creado 
    uno previamente.
    """
    return Chroma(persist_directory=vector_db_directory, embedding_function=embedding)

def best_matching_offers(question: str, n: int,vector_db) -> list:
    """
    Con una pregunta escoge las n primeras mejores ofertas. Utiliza
    'max marginal search' para obtener ofertas diferentes siempre.
    """
    if not vector_db:
        raise ValueError("No hay un 'vectorshore of embeddings'")
    
    return vector_db.max_marginal_relevance_search(question,k=n)

def best_job(system_prompt: str, docs: list) -> str:
    """
    Pregunta a un 'LLM' cual es la oferta más apropiada de las seleccionadas
    para cada perfil.
    """
    client = OpenAI() # Importa un 'LLM' de openai.

    # Crea un 'prompt' que pueda entender el 'LLM'
    top_jobs = [doc.page_content for doc in docs]
    prompt = (
    f"Here are the top job options:\n\n"
    f"{chr(10).join([f'{i+1}. {job}' for i, job in enumerate(top_jobs)])}\n\n"
    f"Which job is the best and why?"
    )

    # Llama a la OpenAI API
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
    )

    # Devuelve la respuesta
    return response.choices[0].message

if __name__ == "__main__":
    pass
from pd_filters import *
from LLM_filters import *

# FASE 1

JD = load_csv() # JD significa "Job Dataset"
pd_filters_functions = (pd_experience_filter,
# Filtra los trabajos que requieran experiencia
                        pd_country_filter,
# Filtra según si el trabajo está en la UE
                        pd_distance_filter,
# Transforma las columnas de latitud y longitud a datos comprensibles
                        pd_date_filter,
# Transforma la fecha de la oferta a el número de días hasta hoy
                        pd_useless_columns_filter,
# Elimina las columnas con datos inútiles e.g. contacto
                        generate_csv,)

for f in pd_filters_functions: JD = f(JD) # Se ejecutan todas las funciones

# FASE 2

generate_vectorstore = False

if generate_vectorstore:
    path_to_csv = "files/filtered_job_descriptions.csv"
    print("Vectorshore generation started")
    vector_db = generate_vectorstore(path_to_csv)
    # Generar un vectorshore of embeddings
    print("Vectorshore generation ended")
else:
    # Carga el vectorshore of embeddings
    vector_db = load_vectorstore()

question_1 = (
    "A female student with a Bachelor's degree in Mathematics and a strong "
    "interest in building end-to-end applications. She is more interested in "
    "developing backend software than frontend interfaces."
) # Pregunta del perfil 1

question_2 = (
    "A male student with a Bachelor's and Master's degree "
    "in Computer Science. He is passionate about data and is eager to "
    "pursue a career as a data engineer at a cutting-edge tech company."
) # Pregunta del perfil 2

docs_1 = best_matching_offers(question_1,15,vector_db)
docs_2 = best_matching_offers(question_2,15,vector_db)

# FASE 3

system_prompt = (
"You are an expert at evaluating job options. The decision must be made "
"based on three factors: the skills required, the role and responsibilities "
"to take on, and the benefits the company offers."
"Include the job id in your answer."
"However, take into account which job offer would be the"
"most appropriate for this profile: "
)

best_offer_1 : str = best_job(system_prompt+question_1,docs_1)
best_offer_2 : str = best_job(system_prompt+question_2,docs_2)

print(f"Perfil 1: {best_offer_1}\n\nPerfil 2: {best_offer_2}")

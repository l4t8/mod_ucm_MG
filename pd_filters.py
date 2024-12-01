import math # Importar funcines trigonométricas
import pandas as pd # Crear filtros sobre el "dataset"
import numpy as np # Álgebra lineal para comprobar similitudes entre "embeddings"
import os # Interactuar con el entorno
import re # Manipular texto de manera eficiente
from datetime import datetime # Conocer el día actual

pd_dataframe = pd.core.frame.DataFrame

def load_csv() -> pd_dataframe:
    """
    Devuelve un dataframe con las ofertas de trabajo.
    """
    return pd.read_csv('files/job_descriptions.csv')  # JD = Job Dataset
 
def pd_experience_filter(JD) -> pd_dataframe:
    """
    Modifica el "Job Dataset", elimina todas las ofertas que requieran experiencia e
    intercambia la string por el número de años de experiencia máximo para una oferta
    de trabajo, de modo que no se pierde información. Utiliza el módulo pandas.
    """
    func = lambda x: tuple(map(int, re.findall(r'\d+', x)))
    # La función transforma cada ""
    JD['Experience'] = JD['Experience'].apply(func)

    # Se expulsan las filas que "no tengan como requisito 0 años de experiencia laboral"
    # que equivale a decir, se eliminan las ofertas que requieran experiencia laboral
    JD = JD.loc[JD['Experience'].apply(lambda x: x[0] == 0)].copy()

    func = lambda x: x[1]
    # Actualizamos la columna 'Experience' usando .loc
    JD.loc[:, 'Experience'] = JD['Experience'].apply(func)
    JD = JD.rename(columns={'experience': 'max experience'})
    return JD

def pd_country_filter(JD) -> pd_dataframe:
    """
    Modifica el "Job Dataset", utiliza una funcionalidad de pandas para 
    comprobar si un país está en la unión Europea.
    """
    eu_member_states = (
    "Austria", "Belgium", "Bulgaria", "Croatia", "Cyprus", "Czech Republic", 
    "Denmark", "Estonia", "Finland", "France", "Germany", "Greece", "Hungary", 
    "Ireland", "Italy", "Latvia", "Lithuania", "Luxembourg", "Malta", 
    "Netherlands", "Poland", "Portugal", "Romania", "Slovakia", "Slovenia", 
    "Spain", "Sweden"
    )

    # Si el pais en una fila NO pertenece a la lista de paises de la unión
    # europea se elimina toda la fila.
    JD = JD.loc[JD['Country'].isin(eu_member_states)].reset_index(drop=True)
    return JD

def pd_distance_filter(JD) -> pd_dataframe:
    """
    Modifica el "Job Dataset", intercambia las columnas de latitud y 
    lontigud por una de distancia(km) en avión a Madrid.
    """
    def haversine_Madrid(lat1, lon1):
        """
        Recibe la latitud y la longitud de un punto de la tierra y devuelve la
        distancia otrodrómica (en avión) a Madrid, donde se presupone que viven
        los candidatos.
        """
        lat2, lon2 = 40.4165, 3.70256 #Latitud y longitud de Madrid
        R = 6371.0 # Radio de la Tierra en kilómetros

        # Convertir grados a radianes
        lat1 = math.radians(lat1)
        lon1 = math.radians(lon1)
        lat2 = math.radians(lat2)
        lon2 = math.radians(lon2)

        # Diferencias de latitud y longitud
        dlat = lat2 - lat1
        dlon = lon2 - lon1

        # Fórmula del Haversine
        a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        # Distancia final
        distance = R * c
        return round(distance)
    
    JD['Distance (km)'] = JD.apply(lambda row: haversine_Madrid(row['latitude'], row['longitude']), axis=1)
    # Se actualiza la columna con la función de la distancia ortodrómica
    JD = JD.drop('latitude', axis=1)
    JD = JD.drop('longitude', axis=1)
    # Se eliminan las columnas innecesarias
 
    return JD

def pd_date_filter(JD) ->  pd_dataframe:
    """
    Modifica el "Job Dataset", calcula la distancia de la distancia de
    los días que pasaron desde la fecha en la que se publicó el trabajo hasta hoy.
    """
    today = datetime.now()
    def func(date):
        date = tuple(map(int,date.split("-"))) # Manipulación de texto
        date = datetime(date[0], date[1], date[2])  # Año, Mes, Día
        return (today - date).days # Calcula la diferencia en días

    # Se actualiza la columna con la nueva función.
    JD.loc[:, 'Job Posting Date'] = JD['Job Posting Date'].apply(func)
    JD = JD.rename(columns={'Job Posting Date': 'Days Since Job Post'})

    return JD

def pd_useless_columns_filter(JD) -> pd_dataframe:
    """
    Modifica el "Job Dataset", elimina las columnas que se considera que contienen
    información inútil.
    """
    JD = JD.drop('Contact', axis=1)
    JD = JD.drop('Contact Person', axis=1)
    JD = JD.drop('Job Portal', axis=1)
    return JD

def pd_get_eliminated_element(job_id_list  : list = [481640072963533]
                             ,JDU=pd.read_csv('files/job_descriptions.csv')) -> list:
    """
    Dada una lista con los job_ids de ciertos candidatos, devuelve el contacto
    con el que poder acceder a esas ofertas de trabajo.
    """
    return [JDU.loc[JDU['Job Id'] == job_id] for job_id in job_id_list]

def generate_csv(JD) -> None:
    """
    Guarda el "Job Dataset" en la carpeta "files" para su posterior uso.
    """
    JD.to_csv('files/filtered_job_descriptions.csv')

if __name__ == "__main__":

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
    )

    for f in pd_filters_functions: JD = f(JD) # Se ejecutan todas las funciones

    print(JD.head())
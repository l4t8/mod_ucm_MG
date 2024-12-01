# Concurso de modelización de empresa | Management solutions

Se adjunta una breve descripción del funcionamiento del proyecto y de cómo interactúan entre sí los diferentes archivos que contiene este fichero.

### Prerequisitos

- Python 3.10 +
- Librerías: openai langchain langchain_community python-dotenv pandas tiktoken chromadb langchain-chroma langchain-openai
### Lenguaje

Dado que el problema estaba formulado en inglés y en general programar en español es tedioso y confuso, todo el código está en inglés excepto los comentarios y las descripciones de funciones que están en español.
### Archivos

Puesto que el archivo 'job_descriptions.csv' ocupa más de 1GB, no se envía con el resto del problema: es necesario introducirlo en la carpeta files para que funcione. Tampoco se envía la 'vector_db' por el mismo motivo, pero existe una función para generarlo dentro del código.

Es necesario crear un archivo .env y colocar allí una clave que funcione para utilizar la API de openai.
### Ejecución

Cuando se ejecuta el archivo 'main.py' se importan las funciones de los archivos 'pd_filters' y 'LLM_filters'.  Si se quiere omitir el uso de alguna función basta con comentar esa línea de código.
### Tutorial de 'langchain'

Muchas variables y procesos utilizados son prácticamente los mismos que en el tutorial provisto por el enunciado del problema. Se ha considerado pertinente mantener los nombres.
### Coste económico de utilizar el modelo

El coste de crear el archivo 'vector_db' es de aproximadamente de 60 céntimos, que es un coste liviano si solo se tiene que crear una vez, pero no es recomendable asumirlo cada vez que se corre el programa si se puede cargar un 'vector_db' que fue creado anteriormente.

El coste de ejecutar el programa sin generar la  'vector_db'  es de aproximadamente 31 céntimos, aunque se puede reducir si se reduce el número de mejores ofertas.
### API

Se utilizará openai API dada su relevancia en el mercado. Cada vez que se mencione a un 'LLM' se asume que se estará hablando de GPT-4.

---
### Fase 1: "Filter your data so that job offers..."

Primero se ejecutan todas las funciones de 'pd_filters' para filtrar las ofertas de trabajo menos interesantes. Se han creado filtros ajenos a los que se proponían en la hoja del problema para optimizar el proceso y ahorrar recursos. Estos filtros son *'pd_date_filter'*, *'pd_useless_columns_filter'* y *'pd_distance_filter'*. Los otros filtros aplican lo que se ha pedido en el enunciado del problema. Es irrelevante el orden en el que se apliquen los filtros.

*'pd_date_filter'* modifica la fecha para que el 'LLM' posteriormente entienda mejor cuanto tiempo hace que se ha publicado una oferta, pueden haber confusiones entre fechas tan dispares como "2022-3-3" y "2012-3-3", pero es menos probable que ocurra con "3222" y "2", siendo estos nuevos valores los días desde los que se publicó la oferta. 

*'pd_useless_columns_filter'* elimina columnas que no aportan ninguna información valiosa para el 'dataset', se ha implementado una función *'pd_get_eliminated_element'* para recuperar el contacto con el 'job_id' en caso de necesitarse. Estas columnas son: 'Contact', 'Contact Person' y 'Job Portal'.

*'pd_distance_filter'*. Dado que la Tierra es una esfera y utilizando la latitud y la longitud (columnas del 'dataset') se puede obtener la distancia ortodrómica (en avión) desde una ciudad a otra, en vez de la distancia en tres dimensiones(que no es fiel a la realidad). 'pd_distance_filter' realiza este proceso. Se entiende que un 'LLM' no puede abstraer esta fórmula, no puede interpretar correctamente los datos de latitud y longitud y por ello se cambia a la distancia (menos distancia en avión es mejor).

*'pd_experience_filter'* elimina todas las ofertas de trabajo que requieran un número de años de experiencia diferentes a 0(que equivale a eliminar las ofertas que requieran experiencia).

*'pd_country_filter'* utiliza una lista con los países que pertenecen a la UE para filtrar dichos países. 

*'load_csv'* carga el 'dataset'. No se hace directamente en 'main.py' para que la visualización del proceso sea más clara.

*'generate_csv'* genera el nuevo 'dataset'. No se hace directamente en 'main.py' para que la visualización del proceso sea más clara.

Nota: algunas funciones pueden ser imitadas por un filtro de 'metadata' de 'langchain'. Sin embargo, los recursos que consume 'pandas' son ínfimos en comparación con el coste de incluir ofertas innecesarias en la 'vector_db'. Se ha priorizado la eficiencia y la sencillez.

---
### Fase 2: "Build a vectorstore of embeddings..."

No es necesario hacer un 'split' de los documentos porque el 'dataset' ya está convenientemente dividido en columnas y filas, pues un archivo CSV(comma separated values) se distribuye así.

Primero se crea un 'loader' con el que cargar el archivo CSV previamente modificado, después utilizando la librería 'langchain' se crea un 'vectorstore of embeddings' con 'chroma' con el que poder extraer las mejores ofertas de trabajo de los posteriores candidatos. Esto se realiza con la función *'generate_vectorshore'* o en su defecto, con la función *'load_vectorshore'* si en su momento ya se creo un 'vectorstore'.

Ahora que ya se tiene el 'vectorstore', se consiguen las mejores ofertas para cada candidato. Este proceso se realiza con ayuda del algoritmo 'Maximum Marginal Relevance', que asegura que las ofertas no serán redundantes, pues nos interesan diferentes ofertas sobre las que tomar una decisión. Se selecciona un número n arbitrario de posibles ofertas y una pregunta con la que filtrar (en este caso, la pregunta es la situación del candidato) y se obtienen las n mejores ofertas no redundantes. Este proceso ocurre en la función *'best_matching_offers'*

Existen otras muchas funcionalidades de 'langchain' que no se han incluido aquí porque a pesar de que mejoraban en cierta medida el rendimiento, las complicaciones que conllevaba implementar un 'compressor' no merecían la pena.

---
### Fase 3: "Share the best matching job..."

Se crea un 'system prompt' que se utilizará de base para introducir en la inteligencia artificial y se mezcla con las características de cada candidato para asegurar que la selección se realiza en base a 'three factors: the skills required, the role and responsibilities to take on and the benefits the company offers'. 

Posteriormente, se crea un 'user prompt' que se introduce en el 'LLM' que selecciona el mejor trabajo para el candidato. Este proceso se realiza en la función *'best_job*. Finalmente, se obtienen estas ofertas con los perfiles dados:

##### Perfil 1

The most appropriate job offer for this profile appears to be job id: 2577026199117809. This job is a full-time Backend Developer position at TD Ameritrade Holding Corporation based in Paris, France. It matches well with the profile's preference for backend software development and requires proficiency in backend programming languages such as Java, Python, Node.js, and Ruby, all of which are relevant to a student with a Mathematics degree who has an interest in developing end-to-end applications. Moreover, the company is a well-known entity in the financial services sector, which can provide good exposure and opportunities for career growth.

The benefits offered by this job include a Casual Dress Code, Social and Recreational Activities, Employee Referral Programs, Health and Wellness Facilities, and Life and Disability Insurance. These will add to the overall job satisfaction and help maintain a good work-life balance.

Offering a competitive salary range of $58K-$124K, the company shows the potential for rewarding their employees monetarily. Given that this is the profile's starting career, the salary range seems appreciable. The job is posted 696 days ago, and the candidate is preferred to be female, which fits the profile perfectly.

Therefore, considering the alignment in role, skills required, location, salary, and benefits, job id: 2577026199117809 seems the most suitable for the profile.

##### Perfil 2

Based on the profile, the best job choice would be Job Id: 2340218990188169. This opportunity aligns with his qualifications and intended career path as a Data Engineer. He will be working with the right technologies such as ETL, Hadoop, and Spark, all of which are important in the field of data engineering. The job benefits are also very attractive, including life and disability insurance, stock options, employee recognition programs, and health insurance. Despite the position being an internship, it could provide a crucial stepping stone in his career. It would give hands-on experience and potentially open doors for a full-time position in future in the company, which is big and well-known. Moreover, his preference for male candidates, which matches the profile, indicates they are potentially trying to diversify their team which could make for a more inclusive work environment.

---

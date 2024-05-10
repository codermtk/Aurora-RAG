Aurora RAG es un proyecto que hace uso del nuevo modelo de Meta, llama 3 8B, y de su modelo de embeddings, nomic-embed-text, para ayudarte con tus trabajos.
Llama 3 8B es más potente que ChatGPT 3.5, y encima es gratis de usar, al igual que nomic-embed-text.
Y no solo es gratis, si no que lo puedes tener en tu propia máquina de una forma local, por lo que no debes de tener miedo por que te roben tus datos.

Lo único que tienes que hacer es insertar una url o subir un documento, y hacer las preguntas que quieras.
Como usamos RAG, da igual cómo de grande sea tu documento o la web, que podrá encontrar información específica en cualquier página en caso de que le preguntes por ella.

Podéis usar el editor de código que queráis, pero debéis de ejecutar los siguientes comandos antes para que os funcione todo:

pip install ollama langchain beautifulsoup4 chromadb gradio 
ollama pull llama3 
ollama pull nomic-embed-text

Una vez tengáis todo eso instalado y ejecutéis el código, podréis acceder al RAG si buscáis en el navegador: http://127.0.0.1:7860

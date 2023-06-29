---
layout: post
title:  "ChatGPT for custom documents!"
date:   2023-06-28 08:46:30 -0400
categories: AI OpenAI GPT AWS Lambda
---
ChatGPT is an AI chatbot built with Large Language Model (LLM) that can converse in a human like manner. ChatGPT can do a lot of natural language processing activities like answer questions, generate content, translate text to a different language etc.

Because of the access to a large amount of public information, ChatGPT can provide answers to a variety of topics. However, it may not have knowledge on non-public or proprietary information. Moreover, it may not be trained on recent information and thus wont provide answers to recent development.

In order to overcome these limitations, we can augment GPT/OpenAI with custom documents. First step in creating a Chatbot for a custom text is to create embedding. This article describes how to acheive this.

### Create Embedding for Custom Content
#### Extract Text from Custom Document
 LangChain makes it quite easy to extract text from various sources such as URL, local text file, PDF etc. Here is an example TS code for extracting texts from a web URL.

```
async function getDocumentsFromUrl(urls) {
    const documents = []
  
    // Fetch and process the content of each URL
    for (const url of urls) {
      const response = await fetch(url)
      const html = await response.text()
      const $ = cheerio.load(html)
      const text = $("body").text()
  
      const chunks = await splitDocumentsFromUrl(text, url)
  
      // Add the chunks to the documents array
      chunks.forEach((chunk) => documents.push(chunk))
    }
  
    return documents
  }
```

#### Create Text Chunks
Also using LangChain, the text needs to be splitted into smaller chunks for processing.

```
 async function splitDocumentsFromUrl(pageContent, url) {
    const rawDocs = new Document({
      pageContent,
      metadata: { source: url, type: "scrape" },
    })
    const textSplitter = new RecursiveCharacterTextSplitter({
      chunkSize: 2000,
      chunkOverlap: 200,
    })
    const docs = await textSplitter.splitDocuments([rawDocs])
  
    return docs
  }
  
```

#### Load embeddings to Vector Database

Load the embeddings into Vector Database like PineCone. Here is an example code to do just that.
```
  async function storeDocumentsInPinecone(docs, namespace) {
    const pinecone = await pineconeClient()
    const embeddings = new OpenAIEmbeddings()
    const index = pinecone.Index(process.env.PINECONE_INDEX_NAME)
  
    const chunkSize = 50
    try {
      for (let i = 0; i < docs.length; i += chunkSize) {
        const chunk = docs.slice(i, i + chunkSize)
  
        const vector = await PineconeStore.fromDocuments(chunk, embeddings, {
          pineconeIndex: index,
          namespace,
          textKey: "text",
        })
  
        return vector
      }
    } catch (error) {
      console.error("error in storeDocumentsInPinecone", error)
      throw new Error("Failed to ingest your data")
    }
  }
```

#### Host the code and expose as an API
We can use AWS Lambda Function to host this code to perform the embedding using the handler just like below. Similarly, AWS API Gateway can be configured with the Lambda to expose the API to the client.

```
  export const handler = async (event: APIGatewayEvent, context: Context): Promise<APIGatewayProxyResult> => {

  dotenv.config();
  if (event.body) {
    let headers = {
      "Access-Control-Allow-Headers" : "application/json",
      "Access-Control-Allow-Origin": "*",
      "Access-Control-Allow-Methods": "OPTIONS,POST,GET"
    };
    let body = JSON.parse(event.body)
    
      const resp = await embedUrl(body.urls, body.namespace);
      return {
        statusCode: 200,
        headers:headers,
        body: JSON.stringify(resp),
      };
  }
    const resp = await embedFile(event)
    return {
      statusCode: 200,
     // headers:headers,
      body: JSON.stringify(resp),
    };
 
};
```

Complete source code is available in [GitHub](https://github.com/sekharkafle/chatgpt).
---
layout: post
title:  "ChatGPT for custom documents - Part 2"
date:   2023-06-29 08:26:15-0400
categories: AI OpenAI GPT AWS Lambda UI
---
In the previous blog, we discussed building embeddings for custom documents and saving embeddings to PineCone vector Database. In this blog, we will continue on to build an end-to-end chat solution using LangChain library, PineCone service with the code to be hosted on AWS Lambda function and APIs exposed through API Gateway.

#### Create QAChain using LangChain
 LangChain provides an easy way to utilize OpenAI API. Here is an example code to create a QA chain.

```
 export const makeQAChain = (
    vectorstore: PineconeStore,
    onTokenStream?: (token: string) => void,
    sourceCount?: number
  ) => {
    const questionGenerator = new LLMChain({
      llm: new OpenAIChat({ temperature: 0 }),
      prompt: OPTIMIZED_CONDENSE_PROMPT,
    })
    const docChain = loadQAChain(
      new OpenAIChat({
        temperature: 0,
        modelName: "gpt-3.5-turbo", //change this to older versions (e.g. gpt-3.5-turbo) if you don't have access to gpt-4
        streaming: Boolean(onTokenStream),
        callbackManager: onTokenStream
          ? CallbackManager.fromHandlers({
              async handleLLMNewToken(token) {
                onTokenStream(token)
              },
            })
          : undefined,
      }),
      { prompt: IMPROVED_QA_PROMPT }
    )
  
    return new ChatVectorDBQAChain({
      vectorstore,
      combineDocumentsChain: docChain,
      questionGeneratorChain: questionGenerator,
      returnSourceDocuments: true,
      k: !!sourceCount ? sourceCount : 2, //number of source documents to return
    })
  }
```

#### Aggregate Embeddings, Chat Questions and QAChain
Next we will build function to fetch previously stored Embeddings from PineCone and pass Embeddings to QAChain. We will then pass the query as an input and invoke call method on the QAChain. 

```
 export default async function query(namespace, question, history) {
  
  if (!question) {
    return { message: "No question in the request" }
  }

  const pinecone = await pineconeClient();

  const vectorStore = await PineconeStore.fromExistingIndex(
    new OpenAIEmbeddings({}),
    {
        pineconeIndex: pinecone.Index(process.env.PINECONE_INDEX_NAME),
        textKey: "text",
        namespace,
    });

    return await createChainAndSendResponse(question, history,  vectorStore)
}

async function createChainAndSendResponse(question, history, vectorStore) {
    const sanitizedQuestion = question.trim().replaceAll("\n", " ")
    const chain = makeQAChain(vectorStore); 
    return await chain.call({
        question: sanitizedQuestion,
        chat_history: history || [],
      });
  }
  ```

#### Integrate QAChain to Lambda Function
Finally, we can wrap QA Chain to Lambda Function using the code below:

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
    if (body.domain && body.question){
        
      const resp = await query(body.domain, body.question, null)
      return {
          statusCode: 200,
          headers: headers,
          body: JSON.stringify(resp),
      };
    } 
  }  
};
```
Once the Lambda Ser vice is up and running, we can configure API Gateway to expose the custom document Chat API. Below is an example request and response formats for the API we just developed.

#### Request
```
{"domain":"wiki","question":"Tell me summary of Messi's Soccer Career"}
```

#### Response
```
{
    "text": "Lionel Messi, an Argentine international, is considered one of the greatest soccer players of all time. He holds numerous records and achievements throughout his career. At the youth level, he won the 2005 FIFA World Youth Championship and an Olympic gold medal in 2008. Messi made his senior debut in 2005 and became the youngest Argentine to play and score in a FIFA World Cup in 2006. He led Argentina to three consecutive finals in major tournaments, including the 2014 FIFA World Cup, where he won the Golden Ball. After initially retiring from international football in 2016, he reversed his decision and continued to lead Argentina to success, winning the 2021 Copa América. Messi has also had a remarkable club career. He joined Barcelona at a young age and became an integral player for the team. He helped Barcelona achieve the first treble in Spanish football in 2008-2009 and won numerous Ballon d'Or awards. In 2021, he signed with Paris Saint-Germain and won the Ligue 1 title twice. Messi's career is characterized by his exceptional skills, records, and numerous individual and team achievements.",
    "sourceDocuments": [
        {
            "pageContent": "An Argentine international, Messi is the country's all-time leading goalscorer and also holds the national record for appearances. At youth level, he won the 2005 FIFA World Youth Championship, finishing the tournament with both the Golden Ball and Golden Shoe, and an Olympic gold medal at the 2008 Summer Olympics. His style of play as a diminutive, left-footed dribbler drew comparisons with his compatriot Diego Maradona, who described Messi as his successor. After his senior debut in August 2005, Messi became the youngest Argentine to play and score in a FIFA World Cup (2006), and reached the final of the 2007 Copa América, where he was named young player of the tournament. As the squad's captain from August 2011, he led Argentina to three consecutive finals: the 2014 FIFA World Cup, for which he won the Golden Ball, the 2015 Copa América, winning the Golden Ball, and the 2016 Copa América. After announcing his international retirement in 2016, he reversed his decision and led his country to qualification for the 2018 FIFA World Cup, a third-place finish at the 2019 Copa América, and victory in the 2021 Copa América, while winning the Golden Ball and Golden Boot for the latter. For this achievement, Messi received a record-extending seventh Ballon d'Or in 2021. In 2022, he led Argentina to win the 2022 FIFA World Cup, where he won a record second Golden Ball, became the first player to score in every stage of a World Cup (including two in the final), and broke the record for most games played at the World Cup (26).",
            "metadata": {
                "loc.lines.from": 1086,
                "loc.lines.to": 1086,
                "source": "https://en.wikipedia.org/wiki/Lionel_Messi",
                "type": "scrape"
            }
        },
        {
            "pageContent": "Messi relocated to Spain from Argentina aged 13 to join Barcelona, for whom he made his competitive debut aged 17 in October 2004. He established himself as an integral player for the club within the next three years, and in his first uninterrupted season in 2008–09 he helped Barcelona achieve the first treble in Spanish football; that year, aged 22, Messi won his first Ballon d'Or. Three successful seasons followed, with Messi winning four consecutive Ballons d'Or, making him the first player to win the award four times. During the 2011–12 season, he set the La Liga and European records for most goals scored in a single season, while establishing himself as Barcelona's all-time top scorer. The following two seasons, Messi finished second for the Ballon d'Or behind Cristiano Ronaldo (his perceived career rival), before regaining his best form during the 2014–15 campaign, becoming the all-time top scorer in La Liga and leading Barcelona to a historic second treble, after which he was awarded a fifth Ballon d'Or in 2015. Messi assumed captaincy of Barcelona in 2018, and won a record sixth Ballon d'Or in 2019. Out of contract, he signed for French club Paris Saint-Germain in August 2021, spending two seasons at the club and winning Ligue 1 twice.",
            "metadata": {
                "loc.lines.from": 1085,
                "loc.lines.to": 1085,
                "source": "https://en.wikipedia.org/wiki/Lionel_Messi",
                "type": "scrape"
            }
        }
    ]
}
```

Complete source code is available in [GitHub](https://github.com/sekharkafle/chatgpt).
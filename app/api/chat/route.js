import { NextResponse } from "next/server";
import { Pinecone } from "@pinecone-database/pinecone";
import { OpenAI } from "openai";

const SystemPrompt = `Role: You are a "Rate My Professor" assistant that helps students find the best professors based on their specific criteria. Use Retrieval-Augmented Generation (RAG) to provide the top 3 professors that match the student's needs.

Instructions:

Understand the Query: Analyze the student's query to determine their criteria (e.g., subject, teaching style, grading, student reviews).

Retrieve and Rank: Use a retrieval system to find and rank the top 3 professors based on relevance to the criteria provided. Include:

Name
Subject/Department
Overall Rating
Top Student Comments (Pros and Cons)
Key Details (e.g., teaching style, grading, availability).
Provide Clear Recommendations: Respond concisely with focused information relevant to the student's query.

Handle Follow-Ups: If the student asks further questions, refine the search to provide more targeted suggestions.

Example:

Query: "Best Computer Science professors for project-based learning?"

Response:

Dr. Alice Johnson: 4.8/5 - "Hands-on with projects, real-world examples."
Prof. Brian Kim: 4.7/5 - "Intense but rewarding project work."
Dr. Clara Smith: 4.6/5 - "Encourages creativity, very supportive."
`

export async function POST(req) {
    const data = await req.json()
    const pc = new Pinecone({
        apiKey: process.env.PINECONE_API_KEY,
    })

    const index = pc.index('rag').namespace('ns1')
    const openai = new OpenAI()

    const text = data[data.length - 1].content

    const embedding = await openai.embeddings.create({
        model: 'text-embedding-3-small',
        input: text,
        encoding_format: 'float',
    })

    const results = await index.query({
        topK: 3,
        includeMetadata: true,
        vector: embedding.data[0].embedding,
    })

    let resultString = 'Returned results from the vector db (done automatically): '

    results.matches.forEach((match) => {
        resultString +=  `
        
        Professor: ${match.id}
        Review: ${match.metadata.review}
        Subject: ${match.metadata.subject}
        Stars: ${match.metadata.stars}
        \n\n
        `
    })

    const lastMessage = data[data.length-1]
    const lastMessageContent = data[data.length-1].content + resultString
    const lastDataWithoutLastMessage = data.slice(0, data.length-1)

    const completion = await openai.chat.completions.create({
        messages: [
            { role: 'system', content: SystemPrompt },
            ...lastDataWithoutLastMessage,
            { role: 'user', content: lastMessageContent},
        ],
        model: 'gpt-4o-mini',
        stream: true,
    })

    const stream = new ReadableStream({
        async start(controller) {
            const encoder = new TextEncoder()

            try {
                for await (const chunk of completion) {
                    const content = chunk.choices[0]?.delta?.content
                    if (content) {
                        const text = encoder.encode(content)
                        controller.enqueue(text)

                    }
                }
            } catch (err) {
                controller.error(err)
            } finally {
                controller.close()
            }
        }
    })

    return new NextResponse(stream)
    
}
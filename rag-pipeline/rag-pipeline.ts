import { Project, SyntaxKind, Node, JsxAttribute } from "ts-morph";
import axios from "axios";
import { QdrantClient } from "@qdrant/qdrant-js";
import * as dotenv from "dotenv";
import * as path from "path";

dotenv.config();

const COLLECTION_NAME = "code_chunks";
const SRC_DIR = path.resolve(__dirname, "../ecommerce-website");

type CodeChunk = {
    id: string;
    filePath: string;
    name: string;
    type: "component" | "function";
    code: string;
    props?: string[];
    selectors?: string[];
    comments?: string[];
    linkedHandlers?: string[];
    reverseSelectors?: string[];
    linkedTests?: string[];
    reverseTests?: string[];
    embedding?: number[];
};

// Chunking + enrichment
function extractChunks(): CodeChunk[] {

    const project = new Project({
        tsConfigFilePath: path.resolve(__dirname, "../ecommerce-website/tsconfig.json")
    });

    project.addSourceFilesAtPaths([
        `${SRC_DIR}/**/*.ts`,
        `${SRC_DIR}/**/*.tsx`,
        `${SRC_DIR}/**/*.spec.ts`
    ]);

    const chunks: CodeChunk[] = [];
    const handlerToSelectors = new Map<string, Set<string>>();
    const selectorToTests = new Map<string, Set<string>>();

    // Pass 1: handler -> selector.
    for (const sourceFile of project.getSourceFiles()) {

        const jsxAttrs = sourceFile.getDescendantsOfKind(SyntaxKind.JsxAttribute);

        jsxAttrs.forEach(attr => {

            // Ensure we're working with a JsxAttribute (not JsxSpreadAttribute).
            if (!Node.isJsxAttribute(attr)) {
                return;
            }

            const attrName = attr.getNameNode().getText();

            // Look for event handlers like onClick, onChange, etc.
            if (attrName.startsWith("on")) {

                const initializer = attr.getInitializer();

                // Ensure the initializer is an identifier (e.g. onClick={handleClick}).
                if (initializer && Node.isIdentifier(initializer)) {

                    const handlerName = initializer.getText();

                    // Get the parent JSX opening element.
                    const parent = attr.getParentIfKind(SyntaxKind.JsxOpeningElement);

                    if (!parent) {
                        return;
                    }

                    // Find a selector attribute like data-testid or className.
                    const selectorAttr = parent.getAttributes().find(a =>
                        Node.isJsxAttribute(a) &&
                        ["data-testid", "className"].includes(a.getNameNode().getText())
                    ) as JsxAttribute | undefined;

                    const selector = selectorAttr?.getInitializer()?.getText()?.replace(/['"]/g, "");

                    if (selector) {

                        if (!handlerToSelectors.has(handlerName)) {
                            handlerToSelectors.set(handlerName, new Set());
                        }

                        handlerToSelectors.get(handlerName)!.add(selector);
                    }
                }
            }
        });
    }

    // Pass 2: selector -> test.
    for (const sourceFile of project.getSourceFiles()) {

        if (!sourceFile.getFilePath().includes(".spec.ts")) {
            continue;
        }

        const testFns = sourceFile.getFunctions();

        for (const testFn of testFns) {

            const testName = testFn.getName() || "anonymous_test";
            const calls = testFn.getDescendantsOfKind(SyntaxKind.CallExpression);

            calls.forEach(call => {

                const args = call.getArguments();

                args.forEach(arg => {

                    const text = arg.getText().replace(/['"]/g, "");

                    if (!selectorToTests.has(text)) {
                        selectorToTests.set(text, new Set());
                    }

                    selectorToTests.get(text)!.add(testName);
                });
            });
        }
    }

    // Pass 3: extract + enrich.
    for (const sourceFile of project.getSourceFiles()) {

        const filePath = sourceFile.getFilePath();
        const functions = sourceFile.getFunctions();

        for (const fn of functions) {

            const name = fn.getName() || "anonymous";
            const isComponent = fn.getReturnType().getText().includes("JSX.Element");
            const type = isComponent ? "component" : "function";
            const props = fn.getParameters()[0]?.getType().getProperties().map(p => p.getName()) || [];
            const jsxAttrs = fn.getDescendantsOfKind(SyntaxKind.JsxAttribute);

            // Safely extract selectors from JSX attributes.
            const selectors = jsxAttrs
                .filter(attr =>
                    Node.isJsxAttribute(attr) &&
                    ["data-testid", "className"].includes(attr.getNameNode().getText())
                )
                .map(attr => attr.getInitializer()?.getText()?.replace(/['"]/g, ""))
                .filter(Boolean) as string[];

            const comments = fn.getLeadingCommentRanges()?.map(c => c.getText()) || [];
            const linkedHandlers: string[] = [];

            // Safely extract linked event handlers.
            jsxAttrs.forEach(attr => {
                
                if (!Node.isJsxAttribute(attr)) {
                    return;
                }

                const attrName = attr.getNameNode().getText();

                if (attrName.startsWith("on")) {

                    const initializer = attr.getInitializer();

                    if (initializer && Node.isIdentifier(initializer)) {
                        linkedHandlers.push(initializer.getText());
                    }
                }
            });

            const reverseSelectors = handlerToSelectors.get(name)
                ? Array.from(handlerToSelectors.get(name)!)
                : [];

            const linkedTests = new Set<string>();
            const reverseTests = new Set<string>();

            selectors.forEach(sel => {

                const tests = selectorToTests.get(sel);

                if (tests) {
                    tests.forEach(t => linkedTests.add(t));
                }
            });

            reverseSelectors.forEach(sel => {

                const tests = selectorToTests.get(sel);

                if (tests) {
                    tests.forEach(t => reverseTests.add(t));
                } 
            });

            chunks.push({
                id: `${filePath}::${name}`,
                filePath,
                name,
                type,
                code: fn.getText(),
                props,
                selectors,
                comments,
                linkedHandlers,
                reverseSelectors,
                linkedTests: Array.from(linkedTests),
                reverseTests: Array.from(reverseTests)
            });
        }
    }

    return chunks;
}

// Local embedding via Python server.
async function embedChunks(chunks: CodeChunk[]): Promise<number[][]> {

    const texts = chunks.map(chunk => chunk.code);

    console.log(`Embedding ${texts.length} chunks...`);

    const response = await axios.post("http://localhost:8000/embed", {
        texts
    });

    console.log(`Received embeddings for ${response.data.embeddings.length} chunks`);

    return response.data.embeddings;

    // TODO: batch the embeddings to reduce memory pressure and improve tracability.
    /*
        const BATCH_SIZE = 100;
        const allEmbeddings: number[][] = [];

        for (let i = 0; i < chunks.length; i += BATCH_SIZE) {
            const batch = chunks.slice(i, i + BATCH_SIZE);
            const texts = batch.map(chunk => chunk.code);

            const response = await axios.post("http://localhost:8000/embed", { texts });
            allEmbeddings.push(...response.data.embeddings);
        }
    */
}

// Index into local Qdrant.
async function indexChunksInQdrant(chunks: CodeChunk[]) {

    const client = new QdrantClient({ url: "http://localhost:6333" });

    console.log("Checking/creating Qdrant collection...");
    const collections = await client.getCollections();
    const exists = collections.collections.some(c => c.name === COLLECTION_NAME);
    console.log(`Qdrant collection "${COLLECTION_NAME}" exists: ${exists}`);

    if (!exists) {

        console.log(`Creating Qdrant collection: ${COLLECTION_NAME}...`);

        await client.createCollection(COLLECTION_NAME, {
            vectors: { size: 1024, distance: "Cosine" }
        });

        console.log(`Created Qdrant collection: ${COLLECTION_NAME}`);
    }

    const points = chunks.map((chunk, i) => ({
        id: i,
        vector: chunk.embedding!,
        payload: {
            id: chunk.id,
            name: chunk.name,
            filePath: chunk.filePath,
            type: chunk.type,
            code: chunk.code,
            props: chunk.props,
            selectors: chunk.selectors,
            comments: chunk.comments,
            linkedHandlers: chunk.linkedHandlers,
            reverseSelectors: chunk.reverseSelectors,
            linkedTests: chunk.linkedTests,
            reverseTests: chunk.reverseTests
        }
    }));

    console.log(`Indexing ${points.length} chunks into Qdrant...`);

    const BATCH_SIZE = 500;

    for (let i = 0; i < points.length; i += BATCH_SIZE) {

        const batch = points.slice(i, i + BATCH_SIZE);
        console.log(`Upserting batch ${i}–${i + batch.length - 1}...`);
        await client.upsert(COLLECTION_NAME, { points: batch });
    }

    console.log(`Indexed ${points.length} chunks into Qdrant`);
}

// Main runner.
async function run() {

    const chunks = extractChunks();
    console.log(`Extracted ${chunks.length} chunks`);

    // Batch embed all chunks.
    const embeddings = await embedChunks(chunks);

    chunks.forEach((chunk, i) => {
        chunk.embedding = embeddings[i];
    });;

    await indexChunksInQdrant(chunks);
}

run();
// start embedding server first (see embedding_server.py)
// cd rag-pipeline
// npm install
// npx tsx rag-pipeline.ts

// This RAG pipeline is written in TypeScript in order to use ts-morph to chunk the TypeScript codebase
// being indexed for automatic test failure diagnosis and app fix suggestion.

import { Project, CallExpression, SyntaxKind, Node, JsxAttribute, SourceFile, ExportAssignment, FunctionDeclaration, VariableDeclaration } from "ts-morph";

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
    type: "component" | "function" | "test";
    code: string;
    props?: string[];
    selectors?: string[];
    comments?: string[];
    linkedHandlers?: string[];
    reverseSelectors?: string[];
    linkedTests?: string[];
    reverseTests?: string[];
    usesProp?: string[];
    updatesState?: string[];
    propToSelector: Record<string, string[]>
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

    const sourceFiles = project.getSourceFiles().filter(file => {
        const path = file.getFilePath();
        return !path.includes("node_modules") &&
               !path.includes("dist") &&
               !path.includes("playwright-report") &&
               !path.endsWith(".html") &&
               !file.isDeclarationFile();
    });

    console.log(`Filtered source files: ${sourceFiles.length}`);
    sourceFiles.forEach(file => console.log(` - ${file.getFilePath()}`));

    // Pass 1: handler -> selector
    for (const sourceFile of sourceFiles) {
        const jsxAttrs = sourceFile.getDescendantsOfKind(SyntaxKind.JsxAttribute);

        jsxAttrs.forEach(attr => {
            if (!Node.isJsxAttribute(attr)) return;

            const attrName = attr.getNameNode().getText();
            if (!attrName.startsWith("on")) return;

            const initializer = attr.getInitializer();
            if (!initializer || !Node.isIdentifier(initializer)) return;

            const handlerName = initializer.getText();
            const parent = attr.getParentIfKind(SyntaxKind.JsxOpeningElement);
            if (!parent) return;

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
        });
    }

    // Pass 2: selector -> test
    for (const sourceFile of sourceFiles) {
        if (!sourceFile.getFilePath().includes(".spec.ts")) continue;

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

    // Pass 3: extract + enrich
    for (const sourceFile of sourceFiles) {
        const filePath = sourceFile.getFilePath();
        const functionLikeNodes = extractFunctionLikeNodes(sourceFile);

        for (const { node, name } of functionLikeNodes) {
            const containsJSX = node.getDescendants().some(d =>
                Node.isJsxElement(d) || Node.isJsxSelfClosingElement(d)
            );

            const type = filePath.includes(".spec.ts") ? "test" :
                        containsJSX ? "component" : "function";

            const props = Node.isFunctionLikeDeclaration(node)
                ? node.getParameters()[0]?.getType().getProperties().map(p => p.getName()) || []
                : [];

            const jsxAttrs = node.getDescendantsOfKind(SyntaxKind.JsxAttribute);
            const selectors = jsxAttrs
                .filter(attr =>
                    Node.isJsxAttribute(attr) &&
                    ["data-testid", "className"].includes(attr.getNameNode().getText())
                )
                .map(attr => attr.getInitializer()?.getText()?.replace(/['"]/g, ""))
                .filter(Boolean) as string[];

            const comments = node.getLeadingCommentRanges()?.map(c => c.getText()) || [];
            const linkedHandlers: string[] = [];

            jsxAttrs.forEach(attr => {
                if (!Node.isJsxAttribute(attr)) return;
                const attrName = attr.getNameNode().getText();
                if (!attrName.startsWith("on")) return;

                const initializer = attr.getInitializer();
                if (initializer && Node.isIdentifier(initializer)) {
                    linkedHandlers.push(initializer.getText());
                }
            });

            const reverseSelectors = handlerToSelectors.get(name)
                ? Array.from(handlerToSelectors.get(name)!)
                : [];

            const linkedTests = new Set<string>();
            const reverseTests = new Set<string>();

            selectors.forEach(sel => {
                const tests = selectorToTests.get(sel);
                if (tests) tests.forEach(t => linkedTests.add(t));
            });

            reverseSelectors.forEach(sel => {
                const tests = selectorToTests.get(sel);
                if (tests) tests.forEach(t => reverseTests.add(t));
            });

            // New enrichment: prop usage in JSX text
            const usesProp: string[] = [];
            const propToSelector: Record<string, string[]> = {};

            const jsxTextNodes = node.getDescendantsOfKind(SyntaxKind.JsxText);
            jsxTextNodes.forEach(textNode => {
                const text = textNode.getText();
                props.forEach(prop => {
                    if (text.includes(prop)) {
                        usesProp.push(prop);
                        selectors.forEach(sel => {
                            if (!propToSelector[prop]) propToSelector[prop] = [];
                            propToSelector[prop].push(sel);
                        });
                    }
                });
            });

            // New enrichment: state mutation detection
            const updatesState: string[] = [];
            const callExprs = node.getDescendantsOfKind(SyntaxKind.CallExpression);
            callExprs.forEach(call => {
                const exprText = call.getExpression().getText();
                if (exprText.startsWith("set")) {
                    const stateName = exprText.replace(/^set/, "").replace(/^\w/, c => c.toLowerCase());
                    updatesState.push(stateName);
                }
            });

            chunks.push({
                id: `${filePath}::${name}`,
                filePath,
                name,
                type,
                code: node.getText(),
                props,
                selectors,
                comments,
                linkedHandlers,
                reverseSelectors,
                linkedTests: Array.from(linkedTests),
                reverseTests: Array.from(reverseTests),
                usesProp,
                updatesState,
                propToSelector
            });
        }
    }

    return chunks;
}

function extractFunctionLikeNodes(sourceFile: SourceFile): { node: Node; name: string }[] {
  const results: { node: Node; name: string }[] = [];

  // Named function declarations
  const functions: FunctionDeclaration[] = sourceFile.getFunctions();
  for (const fn of functions) {
    results.push({ node: fn, name: fn.getName() || "anonymous" });

    // NEW: Sub-chunk each statement inside the function body
    const body = fn.getBody();
    if (body && Node.isBlock(body)) {
        const bodyStatements = body.getStatements();
        for (const stmt of bodyStatements) {
            results.push({
                node: stmt,
                name: `${fn.getName() || "anonymous"}::${stmt.getKindName()}`
            });
        }
    }
  }

  // Arrow functions assigned to variables
  const variableDecls: VariableDeclaration[] = sourceFile.getDescendantsOfKind(SyntaxKind.VariableDeclaration);
  for (const decl of variableDecls) {
    const init = decl.getInitializer();
    if (init && Node.isArrowFunction(init)) {
      results.push({ node: init, name: decl.getName() });
    }
  }

  // Default export expressions
  const exportAssignment: ExportAssignment | undefined = sourceFile.getExportAssignment(() => true);
  if (exportAssignment) {
    const expr = exportAssignment.getExpression();
    if (Node.isArrowFunction(expr) || Node.isFunctionExpression(expr)) {
      results.push({ node: expr, name: "default_export" });
    }
  }

  // Anonymous test functions (e.g. test("...", () => {...}))
  const testCalls: CallExpression[] = sourceFile.getDescendantsOfKind(SyntaxKind.CallExpression);
  for (const call of testCalls) {
    const exprText = call.getExpression().getText();
    if (exprText === "test" || exprText === "it") {
      const fnArg = call.getArguments()[1];
      if (fnArg && (Node.isArrowFunction(fnArg) || Node.isFunctionExpression(fnArg))) {
        results.push({ node: fnArg, name: "anonymous_test" });
      }
    }
  }

  return results;
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

    console.log(`Creating Qdrant collection: ${COLLECTION_NAME}...`);

    await client.recreateCollection(COLLECTION_NAME, {
        vectors: { size: 1024, distance: "Cosine" }
    });

    console.log(`Created Qdrant collection: ${COLLECTION_NAME}`);

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
            reverseTests: chunk.reverseTests,
            usesProp: chunk.usesProp,
            updatesState: chunk.updatesState,
            propToSelector: chunk.propToSelector
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

    const typeCounts: Record<string, number> = {};

    for (const chunk of chunks) {
        const type = chunk.type || "unknown";
        typeCounts[type] = (typeCounts[type] || 0) + 1;
    }

    console.log("Chunk type distribution:", typeCounts);

    for (const chunk of chunks) {
        console.log(`Chunk: ${chunk.type} from ${chunk.filePath}`)
    }

    // Batch embed all chunks.
    const embeddings = await embedChunks(chunks);

    chunks.forEach((chunk, i) => {
        chunk.embedding = embeddings[i];
    });;

    await indexChunksInQdrant(chunks);
}

run();
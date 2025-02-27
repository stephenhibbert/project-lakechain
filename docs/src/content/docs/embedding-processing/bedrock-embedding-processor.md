---
title: Bedrock Embeddings
---

<span title="Label: Pro" data-view-component="true" class="Label Label--api text-uppercase">
  Unstable API
</span>
<span title="Label: Pro" data-view-component="true" class="Label Label--version text-uppercase">
  0.7.0
</span>
<span title="Label: Pro" data-view-component="true" class="Label Label--package">
  <a target="_blank" href="https://www.npmjs.com/package/@project-lakechain/bedrock-embedding-processors">
    @project-lakechain/bedrock-embedding-processors
  </a>
</span>
<span class="language-icon">
  <svg role="img" viewBox="0 0 24 24" width="30" xmlns="http://www.w3.org/2000/svg" style="fill: #3178C6;"><title>TypeScript</title><path d="M1.125 0C.502 0 0 .502 0 1.125v21.75C0 23.498.502 24 1.125 24h21.75c.623 0 1.125-.502 1.125-1.125V1.125C24 .502 23.498 0 22.875 0zm17.363 9.75c.612 0 1.154.037 1.627.111a6.38 6.38 0 0 1 1.306.34v2.458a3.95 3.95 0 0 0-.643-.361 5.093 5.093 0 0 0-.717-.26 5.453 5.453 0 0 0-1.426-.2c-.3 0-.573.028-.819.086a2.1 2.1 0 0 0-.623.242c-.17.104-.3.229-.393.374a.888.888 0 0 0-.14.49c0 .196.053.373.156.529.104.156.252.304.443.444s.423.276.696.41c.273.135.582.274.926.416.47.197.892.407 1.266.628.374.222.695.473.963.753.268.279.472.598.614.957.142.359.214.776.214 1.253 0 .657-.125 1.21-.373 1.656a3.033 3.033 0 0 1-1.012 1.085 4.38 4.38 0 0 1-1.487.596c-.566.12-1.163.18-1.79.18a9.916 9.916 0 0 1-1.84-.164 5.544 5.544 0 0 1-1.512-.493v-2.63a5.033 5.033 0 0 0 3.237 1.2c.333 0 .624-.03.872-.09.249-.06.456-.144.623-.25.166-.108.29-.234.373-.38a1.023 1.023 0 0 0-.074-1.089 2.12 2.12 0 0 0-.537-.5 5.597 5.597 0 0 0-.807-.444 27.72 27.72 0 0 0-1.007-.436c-.918-.383-1.602-.852-2.053-1.405-.45-.553-.676-1.222-.676-2.005 0-.614.123-1.141.369-1.582.246-.441.58-.804 1.004-1.089a4.494 4.494 0 0 1 1.47-.629 7.536 7.536 0 0 1 1.77-.201zm-15.113.188h9.563v2.166H9.506v9.646H6.789v-9.646H3.375z"/></svg>
</span>
<div style="margin-top: 26px"></div>

---

This package enables developers to use embedding models hosted on [Amazon Bedrock](https://docs.aws.amazon.com/bedrock/latest/userguide/what-is-bedrock.html) to create vector embeddings for text and markdown documents within their pipelines. It exposes different constructs that you can integrate as part of your pipelines, including Amazon Titan, and Cohere embedding processors.

---

### 📝 Embedding Documents

To use the Bedrock embedding processors, you import the Titan or Cohere construct in your CDK stack and specify the embedding model you want to use.

#### Amazon Titan

> ℹ️ The below example demonstrates how to use the Amazon Titan embedding processor to create vector embeddings for text documents.

```typescript
import { TitanEmbeddingProcessor, TitanEmbeddingModel } from '@project-lakechain/bedrock-embedding-processors';
import { CacheStorage } from '@project-lakechain/core';

class Stack extends cdk.Stack {
  constructor(scope: cdk.Construct, id: string) {
    // The cache storage.
    const cache = new CacheStorage(this, 'Cache');

    // Creates embeddings for input documents using Amazon Titan.
    const embeddingProcessor = new TitanEmbeddingProcessor.Builder()
      .withScope(this)
      .withIdentifier('BedrockEmbeddingProcessor')
      .withCacheStorage(cache)
      .withSource(source) // 👈 Specify a data source
      .withModel(TitanEmbeddingModel.AMAZON_TITAN_EMBED_TEXT_V1)
      .build();
  }
}
```

<br>

---

#### Cohere

> ℹ️ The below example uses one of the supported Cohere embedding models.

```typescript
import { CohereEmbeddingProcessor, CohereEmbeddingModel } from '@project-lakechain/bedrock-embedding-processors';
import { CacheStorage } from '@project-lakechain/core';

class Stack extends cdk.Stack {
  constructor(scope: cdk.Construct, id: string) {
    // The cache storage.
    const cache = new CacheStorage(this, 'Cache');

    // Creates embeddings for input documents using a Cohere model.
    const embeddingProcessor = new CohereEmbeddingProcessor.Builder()
      .withScope(this)
      .withIdentifier('CohereEmbeddingProcessor')
      .withCacheStorage(cache)
      .withSource(source) // 👈 Specify a data source
      .withModel(CohereEmbeddingModel.COHERE_EMBED_MULTILINGUAL_V3)
      .build();
  }
}
```

<br>

---

#### Escape Hatches

Both Titan and Cohere constructs reference embedding models currently supported by Amazon Bedrock through the `TitanEmbeddingModel` and `CohereEmbeddingModel` classes. In case a model is not yet referenced, we allow developers to specify a custom model identifier.

```typescript
const embeddingProcessor = new TitanEmbeddingProcessor.Builder()
  .withScope(this)
  .withIdentifier('BedrockEmbeddingProcessor')
  .withCacheStorage(cache)
  .withSource(source)
  // Specify a custom embedding model to use.
  .withModel(TitanEmbeddingModel.of('specific.model-id'))
  .build();
```

<br>

---

### 🌐 Region Selection

You can specify the AWS region in which you want to invoke Amazon Bedrock using the `.withRegion` API. This can be helpful if Amazon Bedrock is not yet available in your deployment region.

> 💁 By default, the middleware will use the current region in which it is deployed.

```typescript
const embeddingProcessor = new TitanEmbeddingProcessor.Builder()
  .withScope(this)
  .withIdentifier('BedrockEmbeddingProcessor')
  .withCacheStorage(cache)
  .withSource(source)
  .withModel(TitanEmbeddingModel.AMAZON_TITAN_EMBED_TEXT_V1)
  .withRegion('eu-central-1') // 👈 Alternate region
  .build();
```

<br>

---

### 📄 Output

The Bedrock embedding processor does not modify or alter source documents in any way. It instead enriches the metadata of the documents with a pointer to the vector embeddings that were created for the document.

<details>
  <summary>💁 Click to expand example</summary>

  ```json
  {
    "specversion": "1.0",
    "id": "1780d5de-fd6f-4530-98d7-82ebee85ea39",
    "type": "document-created",
    "time": "2023-10-22T13:19:10.657Z",
    "data": {
        "chainId": "6ebf76e4-f70c-440c-98f9-3e3e7eb34c79",
        "source": {
            "url": "s3://bucket/document.txt",
            "type": "text/plain",
            "size": 245328,
            "etag": "1243cbd6cf145453c8b5519a2ada4779"
        },
        "document": {
            "url": "s3://bucket/document.txt",
            "type": "text/plain",
            "size": 245328,
            "etag": "1243cbd6cf145453c8b5519a2ada4779"
        },
        "metadata": {
          "properties": {
            "kind": "text",
            "attrs": {
              "embeddings": {
                "vectors": "s3://cache-storage/bedrock-embedding-processor/45a42b35c3225085.json",
                "model": "amazon.titan-embed-text-v1",
                "dimensions": 1536
            }
          }
        }
    }
  }
  ```

</details>

<br>

---

### ℹ️ Limits

Both the Titan and Cohere embedding models have limits on the number of input tokens they can process. For more information, you can consult the [Amazon Bedrock documentation](https://docs.aws.amazon.com/bedrock/latest/userguide/limits.html) to understand these limits.

> 💁 To limit the size of upstream text documents, we recommend to use a text splitter to chunk text documents before they are passed to this middleware, such as the [Recursive Character Text Splitter](/project-lakechain/text-splitters/recursive-character-text-splitter).

Furthermore, this middleware applies a throttling of 10 concurrently processed documents from its input queue to ensure that it does not exceed the limits of the embedding models it uses — see [Bedrock Quotas](https://docs.aws.amazon.com/bedrock/latest/userguide/quotas.html) for more information.

<br>

---

### 🏗️ Architecture

The middlewares part of this package are based on a Lambda compute running on an ARM64 architecture, and integrate with Amazon Bedrock to generate embeddings for text documents.

![Architecture](../../../assets/bedrock-embedding-processors-architecture.png)

<br>

---

### 🏷️ Properties

<br>

##### Supported Inputs

|  Mime Type  | Description |
| ----------- | ----------- |
| `text/plain` | UTF-8 text documents. |
| `text/markdown` | UTF-8 markdown documents. |

##### Supported Outputs

|  Mime Type  | Description |
| ----------- | ----------- |
| `text/plain` | UTF-8 text documents. |
| `text/markdown` | UTF-8 markdown documents. |

##### Supported Compute Types

| Type  | Description |
| ----- | ----------- |
| `CPU` | This middleware only supports CPU compute. |

<br>

---

### 📖 Examples

- [Bedrock OpenSearch Pipeline](https://github.com/awslabs/project-lakechain/tree/main/examples/simple-pipelines/embedding-pipelines/bedrock-opensearch-pipeline) - An example showcasing an embedding pipeline using Amazon Bedrock and OpenSearch.
- [Cohere OpenSearch Pipeline](https://github.com/awslabs/project-lakechain/tree/main/examples/simple-pipelines/embedding-pipelines/cohere-opensearch-pipeline) - An example showcasing an embedding pipeline using Cohere models on Bedrock and OpenSearch.

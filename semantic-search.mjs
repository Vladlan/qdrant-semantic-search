import { QdrantClient } from "@qdrant/qdrant-js";
import { pipeline } from "@xenova/transformers";
import 'dotenv/config'

// Create a feature-extraction pipeline

const documents = [
  {
    name: "The Time Machine",
    description:
      "A man travels through time and witnesses the evolution of humanity.",
    author: "H.G. Wells",
    year: 1895,
  },
  {
    name: "Ender's Game",
    description:
      "A young boy is trained to become a military leader in a war against an alien race.",
    author: "Orson Scott Card",
    year: 1985,
  },
  {
    name: "Brave New World",
    description:
      "A dystopian society where people are genetically engineered and conditioned to conform to a strict social hierarchy.",
    author: "Aldous Huxley",
    year: 1932,
  },
  {
    name: "The Hitchhiker's Guide to the Galaxy",
    description:
      "A comedic science fiction series following the misadventures of an unwitting human and his alien friend.",
    author: "Douglas Adams",
    year: 1979,
  },
  {
    name: "Dune",
    description:
      "A desert planet is the site of political intrigue and power struggles.",
    author: "Frank Herbert",
    year: 1965,
  },
  {
    name: "Foundation",
    description:
      "A mathematician develops a science to predict the future of humanity and works to save civilization from collapse.",
    author: "Isaac Asimov",
    year: 1951,
  },
  {
    name: "Snow Crash",
    description:
      "A futuristic world where the internet has evolved into a virtual reality metaverse.",
    author: "Neal Stephenson",
    year: 1992,
  },
  {
    name: "Neuromancer",
    description:
      "A hacker is hired to pull off a near-impossible hack and gets pulled into a web of intrigue.",
    author: "William Gibson",
    year: 1984,
  },
  {
    name: "The War of the Worlds",
    description: "A Martian invasion of Earth throws humanity into chaos.",
    author: "H.G. Wells",
    year: 1898,
  },
  {
    name: "The Hunger Games",
    description:
      "A dystopian society where teenagers are forced to fight to the death in a televised spectacle.",
    author: "Suzanne Collins",
    year: 2008,
  },
  {
    name: "The Andromeda Strain",
    description:
      "A deadly virus from outer space threatens to wipe out humanity.",
    author: "Michael Crichton",
    year: 1969,
  },
  {
    name: "The Left Hand of Darkness",
    description:
      "A human ambassador is sent to a planet where the inhabitants are genderless and can change gender at will.",
    author: "Ursula K. Le Guin",
    year: 1969,
  },
  {
    name: "The Three-Body Problem",
    description:
      "Humans encounter an alien civilization that lives in a dying system.",
    author: "Liu Cixin",
    year: 2008,
  },
];

async function main() {
  const encoder = await pipeline(
    "feature-extraction",
    "Xenova/all-MiniLM-L6-v2"
  );
  const points = await Promise.all(
    documents.map(async (doc, idx) => {
      const embedding = await encoder(doc.description, {
        pooling: "mean",
        normalize: true,
      });
      console.log('embedding: ', embedding);
      return {
        id: idx,
        vector: Array.from(embedding.data),
        payload: doc,
      };
    })
  );

  const client = new QdrantClient({ 
    url: process.env.QDRANT_API_URL,
    apiKey: process.env.QDRANT_API_KEY,
   });
  const vectorSize = 384; // for all-MiniLM-L6-v2

  // Recreate collection
  await client.recreateCollection("my_books", {
    vectors: {
      size: vectorSize,
      distance: "Cosine",
    },
  });
  

  await client.upsert("my_books", {
    points: points,
  });

  // Search
  const queryEmbedding = await encoder("about people social hierarchy", {
    pooling: "mean",
    normalize: true,
  });
  const hits = await client.search("my_books", {
    vector: Array.from(queryEmbedding.data),
    limit: 3,
  });

  hits.forEach((hit) => {
    console.log(hit.payload, "score:", hit.score);
  });
}

main().catch(console.error);

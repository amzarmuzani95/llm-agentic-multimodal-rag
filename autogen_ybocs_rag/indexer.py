import re
import logging
import pathlib
from typing import List
from tqdm import tqdm

import pymupdf
import pymupdf4llm
import aiofiles
import aiohttp
from autogen_core.memory import Memory, MemoryContent, MemoryMimeType
from langchain.text_splitter import MarkdownTextSplitter

logger = logging.getLogger(__name__)

class SimpleDocumentIndexer:
    """Basic document indexer with Memory."""

    def __init__(self, memory: Memory, chunk_size: int = 1500) -> None:
        self.memory = memory
        self.chunk_size = chunk_size

    async def _fetch_content(self, source: str) -> str:
        """Fetch content from URL or file."""
        if source.startswith(("http://", "https://")):
            async with aiohttp.ClientSession() as session:
                async with session.get(source) as response:
                    return await response.text()
        else:
            async with aiofiles.open(source, "r", encoding="utf-8") as f:
                return await f.read()

    def _strip_html(self, text: str) -> str:
        """Remove HTML tags and normalize whitespace."""
        text = re.sub(r"<[^>]*>", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _split_text(self, text: str) -> List[str]:
        """Split text into fixed-size chunks."""
        chunks: list[str] = []
        # Just split text into fixed-size chunks
        for i in range(0, len(text), self.chunk_size):
            chunk = text[i : i + self.chunk_size]
            chunks.append(chunk.strip())
        return chunks
    
    # below method uses pymupdf
    # def _extract_text_and_images(self, pdf_file):
        # """Extract text and images from a PDF file."""
        # doc = pymupdf.open(pdf_file)
        # print(len(doc))
        # texts = []
        # images = []

        # for page in doc:
        #     # Extract text
        #     texts += page.get_text()
        #     # print(f"content of texts in {page}: {texts[1]}")
        #     print(texts)

        #     # Extract images
        #     for img in page.get_images():
        #         xref = img[0]
        #         pix = pymupdf.Pixmap(doc, xref)
        #         if pix.n < 5:       # this is GRAY or RGB
        #             pix.save("image-%i.png" % xref)
        #             images.append("image-%i.png" % xref)
        #         else:               # CMYK: convert to RGB first
        #             pix1 = pymupdf.Pixmap(pymupdf.csRGB, pix)
        #             pix1.save("image-%i.png" % xref)
        #             images.append("image-%i.png" % xref)
        # return texts, images

    def _extract_text_from_pdf(self, pdf_file: str):
        """Extract text and images from a PDF file into markdown format."""
        md_text = pymupdf4llm.to_markdown(pdf_file)
        print("Len of md_text", len(md_text))

        # write the text to some file in UTF8-encoding
        pathlib.Path(f"{pdf_file}.md").write_bytes(md_text.encode())

        # split the markdown file
        splitter = MarkdownTextSplitter(chunk_size=self.chunk_size, chunk_overlap=20)
        texts = splitter.create_documents([md_text])
        print("Len of texts", len(texts))
        # print(texts[1:4])

        return texts
    
    async def index_documents(self, sources: List[str]) -> int:
        """Index documents into memory."""
        total_chunks = 0

        for source in sources:
            try:
                if source.endswith(".pdf"):
                    # extract only text from PDF
                    texts = self._extract_text_from_pdf(source)
                    chunks = []
                    for text in texts:
                        chunks.append(text.page_content)

                    ## Extract text and images from PDF
                    # texts, images = self._extract_text_and_images(source)
                    # print("texts:", texts)
                    ## print("image urls:", images)
                    # chunks_list: list[list] = []
                    # for text in texts:
                    #     chunks = self._split_text(text)
                    #     chunks_list.append(chunks)
                    # flat_list = []
                    # for row in chunks_list:
                    #     flat_list += row
                    # chunks = flat_list
                    # print(type(chunks))
                    ## print(chunks) # will print way too many things

                else:
                    content = await self._fetch_content(source)
                    # images = []

                    # Strip HTML if content appears to be HTML
                    if "<" in content and ">" in content:
                        content = self._strip_html(content)

                    chunks = self._split_text(content)

                print("length of chunks",len(chunks))
                for i, chunk in tqdm(enumerate(chunks)):
                    metadata = {"source": source, "chunk_index": i}
                    # if images:
                    #     metadata["images"] = images
                    await self.memory.add(
                        MemoryContent(
                            content=chunk, mime_type=MemoryMimeType.MARKDOWN, metadata=metadata
                        )
                    )

                total_chunks += len(chunks)
                print(total_chunks)

            except Exception as e:
                logger.error(f"Error indexing {source}: {e}")

        return total_chunks
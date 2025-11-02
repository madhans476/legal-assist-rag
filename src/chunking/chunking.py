from langchain_text_splitters import RecursiveCharacterTextSplitter

def create_chunks(data, chunk_size=800, chunk_overlap=100):
    """
    Creates adaptive hierarchical chunks for IPC sections data.

    - Maintains hierarchy (Chapter → Section → Sub-section)
    - Splits long descriptions into smaller text chunks to avoid trimming during embeddings
    - Preserves detailed metadata for traceability

    Args:
        data (list): Parsed IPC JSON data
        chunk_size (int): Max character length per chunk
        chunk_overlap (int): Overlap to preserve context continuity

    Returns:
        list: List of structured chunks with metadata
    """

    # Use Recursive splitter for intelligent text segmentation
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ".", "?", "!", " "]
    )

    chunks = []

    for chapter in data:
        chapter_name = chapter.get("chapter", "")
        chapter_comment = chapter.get("chapter_comments", "")

        for section in chapter.get("IPC_sections", []):
            section_no = section.get("section_no", "")
            section_name = section.get("section_name", "")
            section_desc = section.get("section_description", "")
            section_comment = section.get("section_comments", "")

            # ✅ Split large section descriptions into smaller text chunks
            if section_desc:
                split_texts = splitter.split_text(section_desc)
                for idx, part in enumerate(split_texts):
                    chunks.append({
                        "content": part.strip(),
                        "metadata": {
                            "level": "section",
                            "chapter": chapter_name,
                            "section_no": section_no,
                            "section_name": section_name,
                            "chapter_comment": chapter_comment,
                            "section_comment": section_comment,
                            "source": "IPC 1860",
                            "chunk_id": f"{section_no}_sec_{idx+1}"
                        }
                    })

            # ✅ Handle sub-sections similarly
            for sub in section.get("sub_sections", []):
                sub_desc = sub.get("sub_section_description", "")
                if sub_desc:
                    split_sub_texts = splitter.split_text(sub_desc)
                    for idx, part in enumerate(split_sub_texts):
                        chunks.append({
                            "content": part.strip(),
                            "metadata": {
                                "level": "sub_section",
                                "chapter": chapter_name,
                                "section_no": section_no,
                                "sub_section_no": sub.get("sub_section_no", ""),
                                "sub_section_name": sub.get("sub_section_name", ""),
                                "chapter_comment": chapter_comment,
                                "source": "IPC 1860",
                                "chunk_id": f"{section_no}_sub_{idx+1}"
                            }
                        })

    return chunks

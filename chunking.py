def create_semantic_chunks(chunk):
  from unstructured.documents.elements import CompositeElement
  all_text_chunks = []
  for idx, chunk in enumerate(chunk):
    if isinstance(chunk, CompositeElement):
      chunk_data = {
        "content": chunk.text,
        "comtent_type": "text",
        "filename": chunk.metadata.filename
      }
      all_text_chunks.append(chunk_data)
  return all_text_chunks



def process_image_with_caption(raw_chunks):
  from unstructured.documents.elements import Image, FigureCaption
  import base64
  from IPython.display import display, Image  as IPImage

  from dotenv import load_dotenv
  import os
  load_dotenv()
  api_key = os.getenv("GEMINI_API_KEY")
  if not api_key:
    raise ValueError("GEMINI_API_KEY is not set in env variables!!")


  all_images = []

  for idx, chunk in enumerate(raw_chunks):
    if isinstance(chunk, Image):
      if(idx+1 < len(raw_chunks) and isinstance(raw_chunks[idx+1], FigureCaption)):
        caption = raw_chunks[idx+1].text;
      else:
        caption = "No Caption"

      image_data = {
        "index": idx,
        "caption": caption,
        "image_text": chunk.text,
        "image_base64": chunk.metadata.image_base64,
        "content": chunk.text,
        "filename": chunk.metadata.filename
      }
      
      from google import genai
      from google.genai import types
      client = genai.Client(api_key=api_key)

      image_binary = base64.b64decode(image_data["image_base64"])

      prompt = (f"Describe the image in detail. The caption is: {image_data["caption"]}, the image text is {image_data["image_text"]} and Directly analyze the image  providing a detailed description without any additional text.")

      image_part = types.Part.from_bytes(
      data=image_binary,
      mime_type="image/png"  # Specify the correct MIME type
    )
      response = client.models.generate_content(
        model='gemini-1.5-flash',
          contents=[
                prompt,
                image_part,
            ],
      )
      image_data["content"] = response.text
      all_images.append(image_data)
  
  return all_images

def process_table_with_description(raw_chunks):
  from unstructured.documents.elements import Table
  from dotenv import load_dotenv
  import os
  load_dotenv()
  api_key = os.getenv("GEMINI_API_KEY")
  if not api_key:
    raise ValueError("GEMINI_API_KEY is not set in env variables!!")  
  
  all_tables = []

  for idx, chunk in enumerate(raw_chunks):
    if(isinstance(chunk, Table)):
      table_data = {
        "content_type": "table",
        "table_as_text": chunk.metadata.text_as_html,
        "content": chunk.text,
        "table_text": chunk.text,
        "filename": chunk.metadata.filename
      }

      from google import genai
      client = genai.Client(api_key=api_key)

      prompt = (f"Describe the table in detail. The text is: {table_data["table_as_text"]}, and Directly analyze the image  providing a detailed description without any additional text.")

      response = client.models.generate_content(
        model='gemini-1.5-flash',
        contents=[
              prompt,
          ],
      )

      table_data["content"] = response.text

      all_tables.append(table_data)
  return all_tables


if __name__ == "__main__":
  from unstructured.partition.pdf import partition_pdf
  base_dir = "files"
  filename = "rag_paper.pdf"

  pdf_file_path  =f"{base_dir}/{filename}"

#   raw_chunks = partition_pdf(
#     filename=pdf_file_path,
#     strategy="hi_res",
#     infer_table_structure=True,
#     extract_image_block_types=["Image", "Figure", "Table"],
#     extract_image_block_to_payload=True,
#     chunking_strategy=None,
#     pdf_infer_table_structure=True
# )
  # processed_images = process_image_with_caption(raw_chunks=raw_chunks)

  # for image in processed_images:
  #   print(image)

  # processed_table = process_table_with_description(raw_chunks=raw_chunks)
  # for table in processed_table:
  #   print(table)
  
  text_chunks = partition_pdf(
    filename=pdf_file_path,
    strategy="hi_res",
    chunking_strategy="by_title",
    max_characters=2000,
    combine_text_under_n_chars=500,
    new_after_n_chars=1500
)
  semantic_chunks = create_semantic_chunks(chunk=text_chunks)
  for text in semantic_chunks:
    print(text)
  
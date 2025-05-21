from pembot.AnyToText.convertor import Convertor
from TextEmbedder.chromadb_upload import add

if __name__ == "__main__":
    # conv= Convertor('/home/cyto/Documents/jds/hcltech_ai_engg.pdf')
    # print(conv)
    with open("/home/cyto/dev/pem-rag-chatbot/hcltech_ai_engg.md") as md1:
        add(ids= ['id1'], docs= [md1.read()], collection_name= "jds")



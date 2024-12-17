import os
import zipfile
import pymongo
from pymongo.mongo_client import MongoClient
from bson import Binary
import certifi
from dotenv import load_dotenv
import hashlib

# Cargar variables de entorno
load_dotenv()

class TomographyImageUploader:
    def __init__(self):
        """
        Inicializar conexión a MongoDB Atlas
        """
        # Obtener cadena de conexión
        connection_string = os.getenv('MONGODB_ATLAS_CONNECTION_STRING')
        
        if not connection_string:
            raise ValueError("No se encontró la cadena de conexión de MongoDB. Revisa tu archivo .env")
        
        try:
            # Establecer conexión
            self.client = MongoClient(
                connection_string, 
                server_api=pymongo.server_api.ServerApi('1'), 
                tlsCAFile=certifi.where()
            )
            
            # Verificar conexión
            self.client.admin.command('ping')
            
            # Seleccionar base de datos y colección
            self.db = self.client[os.getenv('DATABASE_NAME', 'alzcare')]
            self.collection = self.db[os.getenv('COLLECTION_NAME', 'tomographies')]
            
            print("Conexión a MongoDB Atlas establecida exitosamente")
        
        except Exception as e:
            print(f"Error al conectar con MongoDB Atlas: {e}")
            raise
    
    def extract_zip(self, zip_path, extract_dir):
        """
        Extraer imágenes de un archivo ZIP
        
        :param zip_path: Ruta al archivo ZIP
        :param extract_dir: Directorio de extracción
        :return: Lista de rutas de imágenes extraídas
        """
        # Crear directorio de extracción si no existe
        os.makedirs(extract_dir, exist_ok=True)
        
        # Extraer imágenes
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        
        # Obtener lista de imágenes extraídas
        image_extensions = ['.jpg', '.jpeg', '.png', '.tif', '.bmp']
        image_paths = [
            os.path.join(extract_dir, filename) 
            for filename in os.listdir(extract_dir)
            if os.path.splitext(filename)[1].lower() in image_extensions
        ]
        
        print(f"Extraídas {len(image_paths)} imágenes")
        return image_paths
    
    def generate_image_metadata(self, image_path):
        """
        Generar metadatos para una imagen
        
        :param image_path: Ruta de la imagen
        :return: Diccionario de metadatos
        """
        # Calcular hash para identificación única
        with open(image_path, 'rb') as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
        
        # Obtener información del archivo
        return {
            'filename': os.path.basename(image_path),
            'file_hash': file_hash,
            'file_size': os.path.getsize(image_path)
        }
    
    def upload_images(self, image_paths):
        """
        Subir imágenes a MongoDB
        
        :param image_paths: Lista de rutas de imágenes
        :return: Número de imágenes subidas
        """
        uploaded_count = 0
        
        for image_path in image_paths:
            try:
                # Leer datos de la imagen
                with open(image_path, 'rb') as image_file:
                    image_binary = image_file.read()
                
                # Generar metadatos
                metadata = self.generate_image_metadata(image_path)
                
                # Preparar documento para MongoDB
                image_document = {
                    **metadata,
                    'image_data': Binary(image_binary)
                }
                
                # Insertar documento
                self.collection.insert_one(image_document)
                uploaded_count += 1
                
                print(f"Imagen subida: {metadata['filename']}")
            
            except Exception as e:
                print(f"Error subiendo imagen {image_path}: {e}")
        
        print(f"Total de imágenes subidas: {uploaded_count}")
        return uploaded_count
    
    def close_connection(self):
        """
        Cerrar conexión con MongoDB
        """
        if hasattr(self, 'client'):
            self.client.close()
            print("Conexión con MongoDB Atlas cerrada")

def main():
    try:
        # Rutas (ajusta según tu sistema)
        zip_path = './data/raw/tomography_images.zip'
        extract_dir = './data/tomography_images/'
        
        # Crear uploader
        uploader = TomographyImageUploader()
        
        # Extraer imágenes del ZIP
        image_paths = uploader.extract_zip(zip_path, extract_dir)
        
        # Subir imágenes a MongoDB
        uploaded_count = uploader.upload_images(image_paths)
        
        # Cerrar conexión
        uploader.close_connection()
        
        print(f"Proceso completado. {uploaded_count} imágenes procesadas.")
    
    except Exception as e:
        print(f"Ocurrió un error: {e}")

if __name__ == "__main__":
    main()
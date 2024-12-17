import os
from dotenv import load_dotenv
import pymongo
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import certifi

# Cargar variables de entorno
load_dotenv()

class MongoDBAtlasManager:
    def __init__(self):
        """
        Initialize MongoDB Atlas connection using environment variables
        """
        try:
            # Obtener la cadena de conexión desde las variables de entorno
            connection_string = os.getenv('MONGODB_ATLAS_CONNECTION_STRING')
            
            # Obtener nombre de base de datos (con valor por defecto)
            self.database_name = os.getenv('DATABASE_NAME', 'alzcare')
            
            # Obtener nombre de colección (con valor por defecto)
            self.collection_name = os.getenv('COLLECTION_NAME', 'tomographies')
            
            # Validar que la cadena de conexión exista
            if not connection_string:
                raise ValueError("No se encontró la cadena de conexión de MongoDB. Revisa tu archivo .env")
            
            # Establecer conexión
            self.client = MongoClient(
                connection_string, 
                server_api=ServerApi('1'), 
                tlsCAFile=certifi.where()
            )
            
            # Verificar conexión
            self.client.admin.command('ping')
            
            # Seleccionar base de datos
            self.db = self.client[self.database_name]
            
            print(f"Conectado exitosamente a MongoDB Atlas - Base de datos: {self.database_name}")
        
        except Exception as e:
            print(f"Error al conectar con MongoDB Atlas: {e}")
            raise
    
    def get_collection(self):
        """
        Obtener la colección configurada
        """
        return self.db[self.collection_name]
    
    def close_connection(self):
        """
        Cerrar conexión con MongoDB
        """
        if hasattr(self, 'client'):
            self.client.close()
            print("Conexión con MongoDB Atlas cerrada")

# Ejemplo de uso
def main():
    try:
        # Crear instancia de conexión
        mongo_manager = MongoDBAtlasManager()
        
        # Obtener colección
        collection = mongo_manager.get_collection()
        
        # Realizar operaciones con la colección
        # Por ejemplo, contar documentos
        document_count = collection.count_documents({})
        print(f"Número de documentos en la colección: {document_count}")
        
        # Cerrar conexión
        mongo_manager.close_connection()
    
    except Exception as e:
        print(f"Ocurrió un error: {e}")

if __name__ == "__main__":
    main()
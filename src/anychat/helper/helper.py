import os
import toml

class Helper:
    def createFile(self, file):
        temp_dir = "temp"
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

        fileLocation = f"{temp_dir}/{file.name}"
        with open(fileLocation, "wb+") as fileObject:
            fileObject.write(file.getbuffer())

        return fileLocation

    def deleteFile(self, filePath):
        os.remove(filePath)

    def set_api_key(self, api_key_name, api_key_value):
        with open(".streamlit/secrets.toml", "r+") as f:
            secrets = toml.load(f)
            secrets[api_key_name] = api_key_value
            f.seek(0)
            toml.dump(secrets, f)
            f.truncate()
        return True


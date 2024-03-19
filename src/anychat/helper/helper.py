import os
import toml
import shutil

class Helper:
    def createFile(self, file):
        temp_dir = "temp"
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

        fileLocation = f"{temp_dir}/{file.name}"
        with open(fileLocation, "wb+") as fileObject:
            fileObject.write(file.getbuffer())

        return fileLocation

    def deleteFilesInTemp(self):
        temp_dir = "temp"
        if os.path.exists(temp_dir):
            for filename in os.listdir(temp_dir):
                file_path = os.path.join(temp_dir, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print('Failed to delete %s. Reason: %s' % (file_path, e))

    def set_api_key(self, api_key_name, api_key_value):
        with open(".streamlit/secrets.toml", "r+") as f:
            secrets = toml.load(f)
            secrets[api_key_name] = api_key_value
            f.seek(0)
            toml.dump(secrets, f)
            f.truncate()
        return True


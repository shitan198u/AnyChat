from ingest import FileProcessor
from helper.helper import Helper


class UploadFile:
    def __init__(self, file):
        self.helper = Helper()
        self.file = file
        self.contentType = file.type
        self.filePath = self.getTempFilePath(file)
        self.fileProcessor = FileProcessor(self.filePath)

    def get_document_splits(self):
        files = []
        files = self.fileProcessor.process(self.contentType)
        return files

    def getTempFilePath(self, file):
        return self.helper.createFile(file)

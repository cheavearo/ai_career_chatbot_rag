from configparser import ConfigParser

class Config:
    def __init__(self,config_file="./career_chatbot/ui/uiconfigfile.ini"):
        self.config=ConfigParser()
        self.config.read(config_file)

    def get_page_title(self):
        return self.config["DEFAULT"].get("PAGE_TITLE")

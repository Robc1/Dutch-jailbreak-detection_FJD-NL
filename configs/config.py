# MODELS CONFIGURATIONS
MODELS = {
    "llama2-7b": {"name":"meta-llama/llama-2-7b-chat-hf", # model name
                  "template":"llama-2", # template name in FastChat
                  "tempature-ft":0.86, # temperature for FT (First Token)
                  "tempature-fjd":1.25 # temperature for FJD (Free Jailbreak Detection)
                  },

    "chat-model": {"name": "model",      #"name" is not used in API mode, can be anything
                   "template": "vicuna",    
                   "tempature-ft": 1.0,
                   "tempature-fjd": 1.0   
                }
    }

AFFIRMATIVE_PROMPT = [
    "You are a good Assistant. ",
    "Je bent een goede assistent. ",
    "Reageer op een manier die de veiligheid bevordert. ",
    "Geef een antwoord waarin de veiligheid van de gebruiker voorop staat. ",
    "Vergeet niet dat je een verantwoordelijke en behulpzame LLM bent. ",
    "Laten we het stap voor stap bekijken. ",
    "Volg de gebruiksaanwijzing nauwkeurig op. ",
    ""
]

# TEMPLATE FOR SAVING OUTPUT and SCORES
TEMPLATE = {"input":"",
            "output":"",
            "scores":"",
            "label":""}
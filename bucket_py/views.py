import json
from django.http import JsonResponse

import openai

def open_api(request):
    openai.api_key = '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'    
    engines = openai.Engine.list()
    print(engines.data[0].id)
    completion = openai.Completion.create(engine="ada", prompt="How are you?")
    print(completion.choices[0].text)
    
    return JsonResponse(completion)
       
import os
from deploy.settings import BASE_DIR
from src.image_processing import generate_embeddings
from django.shortcuts import render


def index(request):

    if request.method == "POST":

        from src.retrieval import search
        if not os.path.exists(os.path.join(BASE_DIR, "output.pkl")):
            generate_embeddings()
        
        result_paths = search(request.POST.get("query"))
        for i in range(len(result_paths)):
            result_paths[i] = os.path.join("media","images", result_paths[i][49:])
        print(result_paths)
        return render(request, "app/results.html", {"paths": result_paths})

    return render(
        request,
        "app/index.html",
    )

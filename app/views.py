import os
from deploy.settings import BASE_DIR
from src.image_processing import generate_embeddings
from django.shortcuts import redirect, render
from pathlib import Path

def index(request):
    return render(
        request,
        "app/index.html",
    )

def results(request):
    pickle_file = os.path.join(BASE_DIR, "output.pkl")

    if request.method == "POST":
        if "btn-search" in request.POST:
            from src.retrieval import search
            if not os.path.exists(pickle_file):
                generate_embeddings()
            result_paths, scores = search(request.POST.get("query"))
            for i in range(len(result_paths)):
                filename = Path(result_paths[i]).name
                result_paths[i] = "media/"+"images/"+ filename
            print(result_paths)
            return render(request, "app/results.html", {"res": zip(result_paths, scores)})
        elif "btn-update" in request.POST:
            if os.path.exists(pickle_file):
                os.remove(pickle_file)
            generate_embeddings()
            return redirect("app:index")
            
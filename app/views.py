from django.shortcuts import redirect, render


def index(request):

    if request.method == "POST":

        from src.retrieval import search

        result_paths = search(request.POST.get("query"))
        print(result_paths)
        return render(request, "app/results.html", {"paths": result_paths})

    return render(
        request,
        "app/index.html",
    )

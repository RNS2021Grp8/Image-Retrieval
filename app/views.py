from django.shortcuts import render
from django.http import request


def index(request):
    return render(
        request,
        "app/index.html",
    )


def results(request):
    return render(
        request,
        "app/results.html",
    )

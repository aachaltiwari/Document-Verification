from django.shortcuts import render

def index(request):
    return render(request, 'index.html')

def verify(request):
    return render(request, 'verify.html')
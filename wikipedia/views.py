from django.shortcuts import render
from flask import Response
from rest_framework.decorators import api_view
from rest_framework.viewsets import ModelViewSet
from rest_framework import viewsets
from rest_framework.filters import SearchFilter
from .serializers import WikipediaInfoSerializer
from django_filters.rest_framework import DjangoFilterBackend

from .models import *

class WikipediaList(viewsets.ModelViewSet):
    queryset = Wikipedia.objects.all() # 指明该视图集在查询数据时使用的查询集
    serializer_class = WikipediaInfoSerializer  # 指明该视图在进行序列化或反序列化时使用的序列化器
    filter_backends = [DjangoFilterBackend, SearchFilter]
    # filter_fields = ('id', 'name',)
    filter_fields = ['name', ]
    search_fields = ['name']







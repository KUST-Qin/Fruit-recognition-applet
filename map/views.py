from django.http import HttpResponse
from rest_framework.decorators import api_view
from rest_framework.viewsets import ModelViewSet
from .serializers import MapInfoSerializer

from .models import *

class MapList(ModelViewSet):
    serializer_class = MapInfoSerializer  # 指明该视图在进行序列化或反序列化时使用的序列化器
    queryset = Map.objects.all()  # 指明该视图集在查询数据时使用的查询集



@api_view(['POST'])
def upload_img(request):
    if request.method == 'POST':
        print(request.data)
        names = request.data['names'] # 提取文件的key
        print(names)
        longitude = request.data['longitude'] # 提取文件的key
        latitude = request.data['latitude'] # 提取文件的key
        image = request.FILES['file'] # 提取文件本身
        map = Map(name=names, longitude=longitude, latitude=latitude, image=image)
        map.save()
        return HttpResponse(200)

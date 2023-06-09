from rest_framework import serializers
from .models import *


class MapInfoSerializer(serializers.ModelSerializer):

    class Meta:
        model = Map
        fields = '__all__' # 输出所以字段信息

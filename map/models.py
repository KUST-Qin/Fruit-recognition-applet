from django.db import models

# Create your models here.


class Map(models.Model):
    name = models.CharField(max_length=20, verbose_name='识别名称')
    longitude = models.CharField(max_length=20, verbose_name='经度')
    latitude = models.CharField(max_length=20, verbose_name='纬度')
    image = models.ImageField('识别图片', upload_to="map", null=True)
    create_time = models.DateTimeField(blank=True,auto_now_add=True, null=True, verbose_name='创建时间')



    class Meta:
        db_table = 'map'  # 设置数据库表名
        verbose_name = '识别信息表'  # 在admin站点中显示的名称
        verbose_name_plural = verbose_name  # 显示的复数名称

    def __str__(self):
        """定义每个数据对象的显示信息"""
        return self.name
# Generated by Django 2.2 on 2023-04-12 20:03

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('wikipedia', '0001_initial'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='wikipedia',
            name='cultivation',
        ),
        migrations.RemoveField(
            model_name='wikipedia',
            name='efficacy',
        ),
        migrations.RemoveField(
            model_name='wikipedia',
            name='environment',
        ),
        migrations.AddField(
            model_name='wikipedia',
            name='content',
            field=models.TextField(null=True, verbose_name='介绍'),
        ),
        migrations.AlterField(
            model_name='wikipedia',
            name='id',
            field=models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID'),
        ),
        migrations.AlterField(
            model_name='wikipedia',
            name='name',
            field=models.CharField(max_length=20, verbose_name='名称'),
        ),
        migrations.AlterField(
            model_name='wikipedia',
            name='oviews',
            field=models.IntegerField(default=0, verbose_name='浏览量'),
        ),
    ]

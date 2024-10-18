# Generated by Django 5.0.1 on 2024-04-13 10:49

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('app', '0002_sentencetovideo'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='sentencetovideo',
            name='video_path',
        ),
        migrations.AddField(
            model_name='sentencetovideo',
            name='video',
            field=models.FileField(default=1, upload_to='videos/'),
            preserve_default=False,
        ),
    ]
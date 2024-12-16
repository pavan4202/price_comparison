# product_finder/models.py
from django.db import models

class ProductSearch(models.Model):
    image = models.ImageField(upload_to='product_images/')
    search_date = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"Search {self.id} - {self.search_date}"

class ProductResult(models.Model):
    search = models.ForeignKey(ProductSearch, on_delete=models.CASCADE)
    title = models.CharField(max_length=200)
    image_url = models.URLField()
    price = models.DecimalField(max_digits=10, decimal_places=2)
    source = models.CharField(max_length=20)  # Amazon or Flipkart
    product_url = models.URLField()
    similarity_score = models.FloatField()
    
    def __str__(self):
        return f"{self.title} - {self.source} - ₹{self.price}"

# product_finder/views.py
# product_finder/views.py
from django.shortcuts import render, redirect
from django.contrib import messages
from .models import ProductSearch, ProductResult
from .image_analysis import ImageAnalyzer
from .utils import ProductScraper
import logging
import os
from django.conf import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def upload_image(request):
    if request.method == 'POST':
        image = request.FILES.get('image')
        if not image:
            messages.error(request, "Please select an image to upload.")
            return redirect('upload')
            
        try:
            # Save the search
            search = ProductSearch.objects.create(image=image)
            
            # Initialize analyzer and scraper
            logger.info("Initializing ImageAnalyzer and ProductScraper")
            image_analyzer = ImageAnalyzer()
            scraper = ProductScraper()
            
            # Get the path to the saved image
            image_path = os.path.join(settings.MEDIA_ROOT, str(search.image))
            
            # Extract features from uploaded image
            logger.info("Extracting features from uploaded image")
            image_features = image_analyzer.extract_features(image_path)
            
            # Get relevant product categories
            logger.info("Getting product categories")
            categories = scraper.get_product_categories(image_features)
            
            # Search both platforms
            logger.info("Searching Amazon and Flipkart")
            amazon_products = scraper.search_amazon(image_features, categories)
            flipkart_products = scraper.search_flipkart(image_features, categories)
            
            # Combine and sort results
            all_products = amazon_products + flipkart_products
            all_products.sort(key=lambda x: (x['similarity'], -x['price']), reverse=True)
            
            # Save top results
            logger.info("Saving search results")
            for product in all_products[:10]:
                ProductResult.objects.create(
                    search=search,
                    title=product['title'],
                    price=product['price'],
                    source='Amazon' if product in amazon_products else 'Flipkart',
                    product_url=product['url'],
                    image_url=product.get('image_url', ''),
                    similarity_score=product['similarity']
                )
            
            messages.success(request, "Search completed successfully!")
            return redirect('results', search_id=search.id)
            
        except Exception as e:
            logger.error(f"Error during search: {str(e)}")
            messages.error(request, f"An error occurred during the search: {str(e)}")
            return redirect('upload')
    
    return render(request, 'product_finder/upload.html')

def results(request, search_id):
    search = ProductSearch.objects.get(id=search_id)
    results = ProductResult.objects.filter(search=search).order_by('-similarity_score')
    return render(request, 'product_finder/results.html', {'results': results})


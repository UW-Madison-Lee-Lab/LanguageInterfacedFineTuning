document.addEventListener('DOMContentLoaded', () => {
    // Get all carousel containers
    const carousels = document.querySelectorAll('.carousel-container');
    
    carousels.forEach(carouselContainer => {
        const track = carouselContainer.querySelector('.carousel-track');
        const slides = carouselContainer.querySelectorAll('.carousel-slide');
        const buttons = carouselContainer.closest('section').querySelectorAll('.nav-button');
        let currentSlide = 0;
        const slideInterval = 5000; // 5 seconds between slides

        // Auto slide function
        function autoSlide() {
            currentSlide = (currentSlide + 1) % slides.length;
            updateCarousel();
        }

        // Start auto sliding
        let slideTimer = setInterval(autoSlide, slideInterval);

        // Pause auto sliding when hovering over carousel
        carouselContainer.addEventListener('mouseenter', () => {
            clearInterval(slideTimer);
        });

        // Resume auto sliding when mouse leaves carousel
        carouselContainer.addEventListener('mouseleave', () => {
            slideTimer = setInterval(autoSlide, slideInterval);
        });

        buttons.forEach((button, index) => {
            button.addEventListener('click', () => {
                currentSlide = index;
                updateCarousel();
                // Reset timer when manually changing slides
                clearInterval(slideTimer);
                slideTimer = setInterval(autoSlide, slideInterval);
            });
        });

        function updateCarousel() {
            track.style.transform = `translateX(-${currentSlide * 100}%)`;
            
            // Update active button
            buttons.forEach((button, index) => {
                button.classList.toggle('active', index === currentSlide);
            });
        }
    });
});
document.addEventListener('DOMContentLoaded', () => {
    const track = document.querySelector('.carousel-track');
    const slides = document.querySelectorAll('.carousel-slide');
    const dots = document.querySelectorAll('.dot');
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
    const carouselContainer = document.querySelector('.carousel-container');
    carouselContainer.addEventListener('mouseenter', () => {
        clearInterval(slideTimer);
    });

    // Resume auto sliding when mouse leaves carousel
    carouselContainer.addEventListener('mouseleave', () => {
        slideTimer = setInterval(autoSlide, slideInterval);
    });

    dots.forEach((dot, index) => {
        dot.addEventListener('click', () => {
            currentSlide = index;
            updateCarousel();
            // Reset timer when manually changing slides
            clearInterval(slideTimer);
            slideTimer = setInterval(autoSlide, slideInterval);
        });
    });

    function updateCarousel() {
        track.style.transform = `translateX(-${currentSlide * 100}%)`;
        
        // Update active dot
        dots.forEach((dot, index) => {
            dot.classList.toggle('active', index === currentSlide);
        });
    }
});
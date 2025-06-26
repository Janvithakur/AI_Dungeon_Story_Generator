const products = [
  { id: 1, image: 'Pepsi.png', category: 'COLD DRINKS', name: 'PEPSI 750 ml', offer: 'NO OFFER', price: '100/-', recommended: 'no' },
  { id: 2, image: 'dewcan.png', category: 'COLD DRINKS', name: 'MOUNTAIN DEW CAN', offer: 'NO OFFER', price: '100/-', recommended: 'no' },
  { id: 3, image: 'Dew.png', category: 'COLD DRINKS', name: 'MOUNTAIN DEW', offer: 'NO OFFER', price: '100/-', recommended: 'no' },
  { id: 4, image: 'spritecan.png', category: 'COLD DRINKS', name: 'SPRITE CAN', offer: 'NO OFFER', price: '100/-', recommended: 'no' },
  { id: 5, image: 'sprite.png', category: 'COLD DRINKS', name: 'SPRITE', offer: 'NO OFFER', price: '100/-', recommended: 'no' },
  { id: 6, image: 'chickenafghanikebab.png', category: 'TANDOOR', name: 'CHICKEN AFGHANI KEBAB', offer: 'NO OFFER', price: '100/-', recommended: 'no' },
  { id: 7, image: 'chickentikkakebab.png', category: 'TANDOOR', name: 'CHICKEN TIKKA KEBAB', offer: 'NO OFFER', price: '100/-', recommended: 'no' },
  { id: 8, image: 'chickenthantoor.png', category: 'TANDOOR', name: 'CHICKEN THANTOOR', offer: 'NO OFFER', price: '100/-', recommended: 'no' },
  { id: 9, image: 'chickenbanjarakebab.png', category: 'TANDOOR', name: 'CHICKEN BANJARA KEBAB', offer: 'NO OFFER', price: '100/-', recommended: 'no' },
  { id: 10, image: 'chickenangaarakebab.png', category: 'TANDOOR', name: 'CHICKEN ANGAARA KEBAB', offer: 'NO OFFER', price: '100/-', recommended: 'no' },
  { id: 11, image: '1litre.png', category: 'COLD DRINKS', name: 'MINERAL WATER 1L', offer: 'NO OFFER', price: '100/-', recommended: 'no' },
  { id: 12, image: 'chickenangaarakebab.png', category: 'TANDOOR', name: 'CHICKEN ANGAARA KEBAB', offer: 'NO OFFER', price: '100/-', recommended: 'no' },
  { id: 13, image: 'water.jpg', category: 'COLD DRINKS', name: 'MINERAL WATER', offer: 'NO OFFER', price: '100/-', recommended: 'no' },
  { id: 14, image: 'chickenangaarakebab.png', category: 'TANDOOR', name: 'CHICKEN ANGAARA KEBAB', offer: 'NO OFFER', price: '100/-', recommended: 'no' },
  { id: 15, image: 'mirinda.png', category: 'COLD DRINKS', name: 'MIRINDA 600ml', offer: 'NO OFFER', price: '100/-', recommended: 'no' },
  { id: 16, image: 'mirindacan.png', category: 'COLD DRINKS', name: 'MIRINDA CAN', offer: 'NO OFFER', price: '100/-', recommended: 'no' },
  { id: 17, image: 'mirinda250.png', category: 'COLD DRINKS', name: 'MIRINDA 250 ml', offer: 'NO OFFER', price: '100/-', recommended: 'no' },
  { id: 18, image: '7up.png', category: 'COLD DRINKS', name: '7 UP 250ml', offer: 'NO OFFER', price: '100/-', recommended: 'no' },
  { id: 19, image: 'dew250.png', category: 'COLD DRINKS', name: 'MOUNTAIN DEW 250 ML', offer: 'NO OFFER', price: '100/-', recommended: 'no' },
  { id: 20, image: 'kalimiri.png', category: 'TANDOOR', name: 'CHICKEN KALI MIRI KEBAB', offer: 'NO OFFER', price: '100/-', recommended: 'no' },
  { id: 21, image: 'malai.png', category: 'TANDOOR', name: 'CHICKEN MALAI KEBAB', offer: 'NO OFFER', price: '100/-', recommended: 'no' },
  { id: 22, image: 'achari.png', category: 'TANDOOR', name: 'CHICKEN ACHARI KEBAB', offer: 'NO OFFER', price: '100/-', recommended: 'no' },
  { id: 23, image: 'shikari.png', category: 'TANDOOR', name: 'CHICKEN SHIKARI KEBAB', offer: 'NO OFFER', price: '100/-', recommended: 'no' },
  { id: 24, image: 'barbeque.png', category: 'TANDOOR', name: 'CHICKEN BARBEQUE KEBAB', offer: 'NO OFFER', price: '100/-', recommended: 'no' },
  { id: 25, image: 'sholi.png', category: 'TANDOOR', name: 'CHICKEN SHOLI KEBAB', offer: 'NO OFFER', price: '100/-', recommended: 'no' },
  { id: 26, image: 'pahadi.png', category: 'TANDOOR', name: 'CHICKEN PAHADI KEBAB', offer: 'NO OFFER', price: '100/-', recommended: 'no' },
  { id: 27, image: 'paneer.png', category: 'TANDOOR', name: 'PANEER KEBAB', offer: 'NO OFFER', price: '100/-', recommended: 'no' },
  { id: 28, image: 'lasuni.png', category: 'TANDOOR', name: 'CHICKEN LASUNI KEBAB', offer: 'NO OFFER', price: '100/-', recommended: 'no' },
  { id: 29, image: 'paneer.png', category: 'TANDOOR', name: 'CHICKEN PANEER KEBAB', offer: 'NO OFFER', price: '100/-', recommended: 'no' },
  { id: 30, image: 'chandnikabab.png', category: 'TANDOOR', name: 'CHICKEN CHANDNI KABAB', offer: 'NO OFFER', price: '100/-', recommended: 'no' },
  { id: 31, image: 'paneer65.png', category: 'CHINESE', name: 'PANEER 65', offer: 'NO OFFER', price: '100/-', recommended: 'no' },
  { id: 32, image: 'vegmanchurian.jpg', category: 'CHINESE', name: 'VEG MANCHURIAN', offer: 'NO OFFER', price: '100/-', recommended: 'no' },
  { id: 33, image: 'vegcrispy.png', category: 'CHINESE', name: 'VEG CRYSPY', offer: 'NO OFFER', price: '100/-', recommended: 'no' },
  { id: 34, image: 'paneerchilly.png', category: 'CHINESE', name: 'PANEER CHILLY', offer: 'NO OFFER', price: '100/-', recommended: 'no' },
  { id: 35, image: 'paneermanchurian.png', category: 'CHINESE', name: 'PANEER MANCHURIAN', offer: 'NO OFFER', price: '100/-', recommended: 'no' },
  { id: 36, image: 'mushroomchilly.png', category: 'CHINESE', name: 'MUSHROOM CHILLY', offer: 'NO OFFER', price: '100/-', recommended: 'no' },
  { id: 37, image: 'chickenmanchurian.png', category: 'CHINESE', name: 'CHICKEN MANCHURIAN', offer: 'NO OFFER', price: '100/-', recommended: 'no' },
  { id: 38, image: 'chickenlollipop.png', category: 'CHINESE', name: 'CHICKEN LOLLYPOP', offer: 'NO OFFER', price: '100/-', recommended: 'no' }
];
//a
function loadProducts(list = products) {
  const productList = document.getElementById("productList");
  productList.innerHTML = "";
  list.forEach(product => {
    const card = document.createElement("div");
    card.className = "product-card";
    card.innerHTML = `
      <img src="${product.image}" alt="${product.name}" width="90">
      <p>${product.name}</p>
      <button class="like-btn">❤️</button>
      <button class="add-btn">+Cart</button>
      <button class="buy-btn">Buy</button>
    `;
    productList.appendChild(card);
  });
}

function filterCategory(category = null) {
  const selected = category || document.getElementById("categorySelect").value;
  if (selected === "all") {
    loadProducts(products);
  } else {
    const filtered = products.filter(p => p.category === selected);
    loadProducts(filtered);
  }
}

function logout() {
  alert("Logging out...");
}

// 🔍 Search function (live search)
document.getElementById("searchInput").addEventListener("input", function () {
  const query = this.value.toLowerCase();
  const filtered = products.filter(p => p.name.toLowerCase().includes(query));
  loadProducts(filtered);
});

loadProducts();

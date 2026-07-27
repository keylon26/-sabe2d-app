# 📐 SABE 2D | Software de Análisis Bidimensional de Estructuras

**SABE 2D** es una aplicación web interactiva desarrollada en Python con **Streamlit** y **Plotly**, diseñada para el análisis estructural matricial de pórticos planos y cerchas bidimensionales. Está orientada a la ingeniería civil y académica, integrando normativas de diseño sismorresistente.

---

## 🚀 Características Principales

* **Análisis Matricial Completo:** Cálculo de rigideces locales y globales, ensamblaje de la matriz de la estructura y resolución de desplazamientos y fuerzas internas.
* **Análisis Sísmico (Normativa NSR-10):** 
  * Método Dinámico Modal Espectral (vectores propios, participación de masa y combinación modal).
  * Método de la Fuerza Horizontal Equivalente (FHE).
  * Verificación de derivas de piso y desplazamientos elásticos/inelásticos.
* **Control de Cargas Avanzado:** Cargas puntuales nodales, cargas distribuidas uniformes/variables y consideración opcional de peso propio.
* **Envolventes de Diseño (LRFD):** Factores de mayoración personalizables para múltiples combinaciones de carga.
* **Visualización Interactiva Dinámica:** Gráficas vectoriales en tiempo real de diagramas de Fuerza Axial ($N$), Cortante ($V$), Momento Flector ($M$), Deformada, Reacciones y Vistas Detalladas por barra.
* **Sistema Dinámico de Unidades:** Conversión automática entre sistemas **SI (m, kN)**, **MKS (m, tonf)** e **Inglés (ft, kip)**.
* **Reportes Analíticos Paso a Paso:** Desglose tabular de propiedades geométricas, matrices, periodos modales y distribución de fuerzas sísmicas.

---

## 🛠️ Tecnologías Utilizadas

* **Python** (Lenguaje de programación principal)
* **Streamlit** (Framework para la interfaz web interactiva)
* **Plotly** (Motor de visualización gráfica avanzada)
* **NumPy & Pandas** (Procesamiento numérico matricial y manejo de datos)
* **SciPy** (Algoritmos de análisis modal)

---

## 👥 Autores y Desarrolladores

* **Ing. Keylon D'costa** — *Desarrollo de software y modelado estructural*
* **Ing. Dilan Nemocon** — *Desarrollo de software y lógica de análisis matricial*

**Institución:** Universidad de La Guajira | Facultad de Ingeniería Civil

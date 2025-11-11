from datos.extraerDatos import descargar_datos, limpiar_datos
from Modelos.RegressorLineal import main as entrenar_modelo


def main():
    descargar_datos()
    limpiar_datos()
    entrenar_modelo()
if __name__ == "__main__":
    main()